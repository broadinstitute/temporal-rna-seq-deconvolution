"""Main deconvolution functionality"""


import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import math
import pandas as pd
import time
import logging
from ternadecov.dataset import DeconvolutionDataset
from ternadecov.parametrization import (
    DeconvolutionDatatypeParametrization,
    TimeRegularizedDeconvolutionModelParametrization,
)
from ternadecov.trajectories import (
    VGPTrajectoryModule,
    BasicTrajectoryModule,
    NonTrajectoryModule,
)
from ternadecov.stats_helpers import NegativeBinomialAltParam

# Indices:
# - c cell type
# - g genes
# - m samples
# - k deformation polynomial degree


def generate_batch(
    dataset: DeconvolutionDataset, device: torch.device, dtype: torch.dtype
):
    """Generate a full training batch
    
    :param dataset: DeconvolutionDataset
    :param device: torch device
    :param dtype: torch dataset
    """

    return {
        "x_mg": dataset.bulk_raw_gex_mg.clone().detach().to(device).type(dtype),
        "t_m": torch.tensor(dataset.dpi_time_m, device=device, dtype=dtype),
    }


_TRAJECTORY_MODEL_TYPES = {"polynomial", "gp", "nontrajectory"}


class TimeRegularizedDeconvolutionModel:
    """Main deconvolution class"""

    def __init__(
        self,
        dataset: DeconvolutionDataset,
        types: DeconvolutionDatatypeParametrization,
        use_betas: bool = True,
        trajectory_model_type: str = "polynomial",
        hyperparameters=None,
        trajectory_hyperparameters=None,
        **kwargs,
    ):
        """Initializer for TimeRegularizedDeconvolutionModel
        
        :param self:
        :param dataset:
        :param types:
        :param use_betas:
        :param trajectory_model_type:
        :param hyperparameters:
        :param trajectory_hyperparameters:
            See below

        :Keyword Arguments:
            * basis_functions (``str``)--
              set of basis functions
            * polynomial_degree (``int``)--
                polynomial degree for legendre and regular polynomials
            
        """

        if hyperparameters is None:
            logging.warning("Model parametrization not provided, using default")
            hyperparameters = TimeRegularizedDeconvolutionModelParametrization()

        self.dataset = dataset

        self.device = types.device
        self.dtype = types.dtype

        self.use_betas = use_betas
        self.trajectory_model_type = trajectory_model_type

        self.init_posterior_global_scale_factor = (
            hyperparameters.init_posterior_global_scale_factor
        )

        # hyperparameters
        self.log_beta_prior_scale = hyperparameters.log_beta_prior_scale
        # self.tau_prior_scale = hyperparameters.tau_prior_scale
        self.log_phi_prior_loc = hyperparameters.log_phi_prior_loc
        self.log_phi_prior_scale = hyperparameters.log_phi_prior_scale

        if trajectory_model_type == "polynomial":
            if trajectory_hyperparameters is not None:
                raise ValueError()

            self.population_proportion_model = BasicTrajectoryModule(
                basis_functions=kwargs["basis_functions"],
                polynomial_degree=kwargs["polynomial_degree"],
                num_cell_types=self.dataset.num_cell_types,
                num_samples=self.dataset.num_samples,
                init_posterior_global_scale_factor=self.init_posterior_global_scale_factor,
                device=self.device,
                dtype=self.dtype,
            )
        elif trajectory_model_type == "gp":
            if trajectory_hyperparameters is None:
                raise ValueError(
                    "Trajectory model GP requires trajectory_hyperparameters to be set"
                )

            self.population_proportion_model = VGPTrajectoryModule(
                xi_mq=self.dataset.t_m[..., None].contiguous(),
                num_cell_types=self.dataset.num_cell_types,
                init_posterior_global_scale_factor=self.init_posterior_global_scale_factor,
                parametrization=trajectory_hyperparameters,
                device=self.device,
                dtype=self.dtype,
            )
        elif trajectory_model_type == "nontrajectory":
            self.population_proportion_model = NonTrajectoryModule(
                num_cell_types=self.dataset.num_cell_types,
                num_samples=self.dataset.num_samples,
                device=self.device,
                dtype=self.dtype,
            )

        else:
            raise ValueError

        self.log_beta_posterior_scale = hyperparameters.log_beta_posterior_scale
        # self.tau_posterior_scale = hyperparameters.tau_posterior_scale
        self.log_phi_posterior_loc = hyperparameters.log_phi_posterior_loc
        self.log_phi_posterior_scale = hyperparameters.log_phi_posterior_scale

        # cache useful tensors
        self.w_hat_gc = torch.tensor(
            self.dataset.w_hat_gc, device=self.device, dtype=self.dtype
        )

    def model(
        self, x_mg: torch.Tensor, t_m: torch.Tensor,
    ):
        """Main model

        :param self: instance of Object
        :param x_mg: gene expression
        :param t_m: obseration time
        """

        # sample log_phi_g
        log_phi_g = pyro.sample(
            "log_phi_g",
            dist.Normal(
                loc=self.log_phi_prior_loc
                * torch.ones(
                    (self.dataset.num_genes,), device=self.device, dtype=self.dtype
                ),
                scale=self.log_phi_prior_scale
                * torch.ones(
                    (self.dataset.num_genes,), device=self.device, dtype=self.dtype
                ),
            ).to_event(1),
        )
        assert log_phi_g.shape == (self.dataset.num_genes,)

        # sample log_beta_g
        log_beta_g = pyro.sample(
            "log_beta_g",
            dist.Normal(
                loc=torch.zeros(
                    (self.dataset.num_genes,), device=self.device, dtype=self.dtype
                ),
                scale=self.log_beta_prior_scale
                * torch.ones(
                    (self.dataset.num_genes,), device=self.device, dtype=self.dtype
                ),
            ).to_event(1),
        )
        assert log_beta_g.shape == (self.dataset.num_genes,)

        # calculate useful derived variables
        beta_g = log_beta_g.exp()
        phi_g = log_phi_g.exp()

        # Get normalized w_gc
        if self.use_betas:
            unnorm_w_gc = self.w_hat_gc * beta_g[:, None]
        else:
            unnorm_w_gc = self.w_hat_gc

        w_gc = unnorm_w_gc / unnorm_w_gc.sum(0)

        # get the prior cell populations from the trajectory module
        cell_pop_mc = self.population_proportion_model.model(
            xi_mq=t_m[..., None].contiguous()
        )

        # calculate mean gene expression
        mu_mg = x_mg.sum(-1)[:, None] * torch.matmul(
            cell_pop_mc, w_gc.transpose(-1, -2)
        )

        with pyro.plate("batch"):
            # observe gene expression
            # todo: sample specific phi?
            pyro.sample(
                "x_mg",
                NegativeBinomialAltParam(mu=mu_mg, phi=phi_g[None, :]).to_event(1),
                obs=x_mg,
            )

    def guide(self, x_mg: torch.Tensor, t_m: torch.Tensor):
        """Main guide
        
        :param self: instance of object
        :param x_mg: expression matrix
        :param t_m: times
        
        :return: posterior draw
        """

        # variational parameters for log_phi_g
        log_phi_posterior_loc_g = pyro.param(
            "log_phi_posterior_loc_g",
            self.log_phi_posterior_loc
            * torch.ones(
                (self.dataset.num_genes,), device=self.device, dtype=self.dtype
            ),
        )

        # variational parameters for log_beta_g
        log_beta_posterior_loc_g = pyro.param(
            "log_beta_posterior_loc_g",
            torch.zeros(
                (self.dataset.num_genes,), device=self.device, dtype=self.dtype
            ),
        )

        # posterior sample statements
        log_phi_g = pyro.sample(
            "log_phi_g", dist.Delta(v=log_phi_posterior_loc_g).to_event(1)
        )

        log_beta_g = pyro.sample(
            "log_beta_g", dist.Delta(v=log_beta_posterior_loc_g).to_event(1)
        )

        # get the posterior cell populations from the trajectory module
        cell_pop_mc = self.population_proportion_model.guide(
            xi_mq=t_m[..., None].contiguous()
        )

        return {
            "log_phi_posterior_loc_g": log_phi_posterior_loc_g,
            "log_beta_posterior_loc_g": log_beta_posterior_loc_g,
            "log_phi_g": log_phi_g,
            "log_beta_g": log_beta_g,
            "cell_pop_mc": cell_pop_mc,
        }

    def fit_model(
        self,
        n_iters=3000,
        log_frequency=100,
        verbose=True,
        clear_param_store=True,
        keep_param_store_history=False,
    ):
        """Iteratively fit the mode
        
        :param self: instance of object
        :param n_inters: number of iterations to execute
        :param log_frequency: log frequncy (in iterations)
        :param verbose: verbosity flat
        :param clear_param_store: flag to clear parameter store before starting iterations
        :param keep_param_store_history: flag to keep full parameter store copies during learning (warning: high memory consumption)
        
        """
        if clear_param_store:
            pyro.clear_param_store()

        optim = pyro.optim.Adam({"lr": 1e-3})

        self.loss_hist = []
        self.param_store_hist = []

        svi = SVI(model=self.model, guide=self.guide, optim=optim, loss=Trace_ELBO())

        start_time = time.time()
        for i_iter in range(n_iters):
            batch_dict = generate_batch(self.dataset, self.device, self.dtype)

            loss = svi.step(**batch_dict)
            self.loss_hist.append(loss)

            if keep_param_store_history:
                param_store = pyro.get_param_store()
                self.param_store_hist.append(
                    {
                        k: v.detach().float().cpu().clone()
                        for k, v in param_store.items()
                    }
                )

            if verbose:
                if i_iter % log_frequency == 0:

                    print(
                        f"[step: {i_iter}, time: {math.ceil(time.time() - start_time)} s ] loss: {self.loss_hist[-1]:.2f}"
                    )

    def sample_composition_default(self):
        """Return the sample composition in a pandas DataFrame
        
        :param self: instance of object
        :return: return the current sample composition in pandas dataframe format
        """

        cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        col_sample = []
        col_celltype = []
        col_proportion = []
        for i_0 in range(cell_pop_mc.shape[0]):
            for i_1 in range(cell_pop_mc.shape[1]):
                col_sample.append(self.dataset.bulk_sample_names[i_0])
                col_celltype.append(self.dataset.cell_type_str_list[i_1])
                col_proportion.append(cell_pop_mc[i_0, i_1].item())
        return pd.DataFrame(
            {
                "col_sample": col_sample,
                "col_celltype": col_celltype,
                "col_proportion": col_proportion,
            }
        )

    def write_sample_compositions(self, csv_filename, ignore_hypercluster=False):
        """Write sample composition to csv file
        
        :param self: instance of object
        :param csv_filename: filename to save the results to
        :param ignore_hypercluster: Flag to ignore hyperclustering if present
        """

        if self.dataset.is_hyperclustered and not ignore_hypercluster:
            raise NotImplementedError
        else:
            self.write_sample_composition_default(csv_filename)

    def write_sample_composition_default(self, csv_filename):
        """Write sample composition proportions to csv file

        :param self: instance of object
        :param csv_filename: filename of csv file to write to
        """

        composition_df = self.sample_composition_default()
        composition_df.to_csv(csv_filename)
