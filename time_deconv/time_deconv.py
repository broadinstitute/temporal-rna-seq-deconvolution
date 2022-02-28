import numpy as np
import matplotlib
import matplotlib.pyplot
from torch.distributions import constraints
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from typing import List, Dict
import pyro.distributions as dist
import anndata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import math
import tqdm
import copy
from matplotlib.pyplot import cm
import pandas as pd
import seaborn as sns
import time
import scanpy as sc
from time_deconv.stats_helpers import *
from time_deconv.time_deconv_simulator import *
from time_deconv.stats_helpers import *
from time_deconv.hypercluster import *
from time_deconv.dataset import *
from time_deconv.trajectories import *

# Indices:
# - c cell type
# - g genes
# - m samples
# - k deformation polynomial degree


class TimeRegularizedDeconvolution:
    def __init__(
        self,
        dataset: DeconvolutionDataset,
        basis_functions: str,
        polynomial_degree: int,
        device: torch.device,
        dtype: torch.dtype,
        use_betas: bool = True,
    ):

        self.dataset = dataset
        self.device = device
        self.dtype = dtype
        self.use_betas = use_betas

        self.init_posterior_global_scale_factor = 0.05

        # hyperparameters
        self.log_beta_prior_scale = 1.0
        self.log_r_prior_scale = 1.0
        self.tau_prior_scale = 1.0
        self.log_phi_prior_loc = -5.0
        self.log_phi_prior_scale = 1.0

        self.population_proportion_model = BasicTrajectoryModule(
            basis_functions=basis_functions,
            polynomial_degree=polynomial_degree,
            num_cell_types=self.dataset.num_cell_types,
            num_samples=self.dataset.num_samples,
            init_posterior_global_scale_factor=self.init_posterior_global_scale_factor,
            device=device,
            dtype=dtype,
        )

        #####################################################
        ## Prior
        #####################################################

        #####################################################
        ## Posterior
        #####################################################

        self.log_beta_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.log_r_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.tau_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.log_phi_posterior_loc = -5.0
        self.log_phi_posterior_scale = 0.1 * self.init_posterior_global_scale_factor

        self.dirichlet_alpha_posterior = (
            self.init_posterior_global_scale_factor * np.ones((1,))
        )

        # cache useful tensors
        self.w_hat_gc = torch.tensor(self.dataset.w_hat_gc, device=device, dtype=dtype)

    def model(
        self, x_mg: torch.Tensor, t_m: torch.Tensor,
    ):
        """Main model

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

        cell_pop_mc = self.population_proportion_model.model(t_m=t_m)

        # calculate mean gene expression
        mu_mg = x_mg.sum(-1)[:, None] * torch.matmul(
            cell_pop_mc, w_gc.transpose(-1, -2)
        )

        with pyro.plate("batch"):
            # observe gene expression
            pyro.sample(
                "x_mg",
                NegativeBinomialAltParam(mu=mu_mg, phi=phi_g[None, :]).to_event(
                    1
                ),  # sample specific phi?
                obs=x_mg,
            )

    def delta_guide(self, x_mg: torch.Tensor, t_m: torch.Tensor):
        """Simple delta guide"""

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

        self.population_proportion_model.guide()

    def fit_model(
        self,
        n_iters=3000,
        log_frequency=100,
        verbose=True,
        clear_param_store=True,
        keep_param_store_history=True,
    ):
        if clear_param_store:
            pyro.clear_param_store()

        ## TODO: bring these out
        optim = pyro.optim.Adam({"lr": 1e-3})

        self.loss_hist = []
        self.param_store_hist = []

        svi = SVI(
            model=self.model, guide=self.delta_guide, optim=optim, loss=Trace_ELBO()
        )

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

    def plot_loss(self):
        """Plot the losses during training"""
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(self.loss_hist)
        ax.set_title("Losses")
        ax.set_xlabel("iteration")
        ax.set_ylabel("ELBO Loss")

        return ax

    def calculate_composition_trajectories(self, n_intervals=100, return_vals=False):
        self.population_proportion_model.calculate_composition_trajectories(
            self.dataset, n_intervals=100, return_vals=False
        )

    def get_composition_trajectories(self):
        """Return the composition trajectories"""
        return self.population_proportion_model.calculated_trajectories

    def plot_composition_trajectories(self, show_hypercluster=False):
        """Plot the composition trajectories"""

        if self.dataset.is_hyperclustered and not show_hypercluster:
            fig, ax = matplotlib.pyplot.subplots()
            ax.plot(
                self.calculated_trajectories["true_times_z"],
                self.calculated_trajectories["summarized_composition_rt"].T,
            )
            ax.set_title("Predicted cell proportions")
            ax.set_xlabel("Time")

            labels = []

            r = self.calculated_trajectories["toplevel_cell_map"]
            map_r = {r[k]: k for k in r}
            for i in range(len(map_r)):
                labels.append(map_r[i])
            ax.legend(labels, loc="best", fontsize="small")
        else:
            fig, ax = matplotlib.pyplot.subplots()
            ax.plot(
                self.population_proportion_model.calculated_trajectories[
                    "true_times_z"
                ],
                self.population_proportion_model.calculated_trajectories[
                    "norm_comp_tc"
                ],
            )
            ax.set_title("Predicted cell proportions")
            ax.set_xlabel("Time")
            ax.legend(self.dataset.cell_type_str_list, loc="best", fontsize="small")

    def plot_phi_g_distribution(self):
        """Plot the distribution of phi_g"""
        phi_g = pyro.param("log_phi_posterior_loc_g").clone().detach().exp().cpu()
        fig, ax = matplotlib.pyplot.subplots()
        ax.hist(phi_g.numpy(), bins=100)
        ax.set_xlabel("$\phi_g$")
        ax.set_ylabel("Counts")

        return ax

    def plot_beta_g_distribution(self):
        """Plot distribution of beta_g"""
        beta_g = pyro.param("log_beta_posterior_loc_g").clone().detach().exp().cpu()
        fig, ax = matplotlib.pyplot.subplots()
        ax.hist(beta_g.numpy(), bins=100)
        ax.set_xlabel("$beta_g$")
        ax.set_ylabel("Counts")

        return ax

    def plot_sample_compositions_scatter(
        self, figsize=(16, 9), ignore_hypercluster=False
    ):
        """Plot a facetted scatter plot of the individual sample compositions

        :param figsize: tuple of size 2 with figure size information
        """
        if self.dataset.is_hyperclustered and not ignore_hypercluster:
            self.plot_sample_compositions_scatter_hyperclustered(figsize=figsize)
        else:
            self.plot_sample_compositions_scatter_default(figsize=figsize)

    def plot_sample_compositions_scatter_default(self, figsize):
        """Plot a facetted scatter plot of the individual sample compositions for regular processing

        :param figsize: tuple of size 2 with figure size information
        """
        t_m = self.dataset.t_m.clone().detach().cpu()
        cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        sort_order = torch.argsort(self.dataset.t_m)

        n_cell_types = cell_pop_mc.shape[1]

        n_rows = math.ceil(math.sqrt(n_cell_types))
        n_cols = math.ceil(n_cell_types / n_rows)

        fig, ax = matplotlib.pyplot.subplots(n_rows, n_cols, figsize=figsize)

        for i in range(cell_pop_mc.shape[1]):
            r_i = int(i // n_rows)
            c_i = int(i % n_rows)

            ax[c_i, r_i].scatter(
                t_m[sort_order] * self.dataset.time_range + self.dataset.time_min,
                cell_pop_mc[sort_order, i].clone().detach().cpu(),
                color=cm.tab10(i),
            )
            ax[c_i, r_i].set_title(self.dataset.cell_type_str_list[i])

        matplotlib.pyplot.tight_layout()

        return ax

    def plot_sample_compositions_scatter_hyperclustered(self, figsize):
        """Plot a facetted scatter plot of the individual sample compositions for hyperclustered processing

        :param figsize: tuple of size 2 with figure size information
        """

        assert self.dataset.is_hyperclustered

        t_m = self.dataset.t_m.clone().detach().cpu()
        cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        sort_order = torch.argsort(self.dataset.t_m)

        ## Summarise cell_pop_mc to the high-level clusters
        n_top_level_clusters = len(
            set(self.dataset.hypercluster_results["cluster_map"].values())
        )
        n_low_level_clusters = len(
            set(self.dataset.hypercluster_results["cluster_map"].keys())
        )
        # k is index for  highlevel clusters
        cell_pop_summarized_mk = torch.zeros(
            (cell_pop_mc.shape[0], n_top_level_clusters)
        )

        # Low level cluster names
        low_cell_type_str_list = self.dataset.cell_type_str_list
        toplevel_cell_map = self.calculated_trajectories["toplevel_cell_map"]
        high_cell_type_str_list = list(toplevel_cell_map.keys())
        low_to_high_clustermap = self.dataset.hypercluster_results["cluster_map"]

        index = torch.zeros((n_low_level_clusters,), dtype=torch.int64)

        for i_llc in range(n_low_level_clusters):
            llc_name = low_cell_type_str_list[i_llc]
            hlc_name = low_to_high_clustermap[llc_name]
            i_hlcc = high_cell_type_str_list.index(hlc_name)
            index[i_llc] = i_hlcc

        cell_pop_summarized_mk.index_add_(1, index, cell_pop_mc)

        print(f"cell_pop_summarized_mk shape: {cell_pop_summarized_mk.shape}")

        n_cell_types = cell_pop_summarized_mk.shape[1]

        n_rows = math.ceil(math.sqrt(n_cell_types))
        n_cols = math.ceil(n_cell_types / n_rows)

        fig, ax = matplotlib.pyplot.subplots(n_rows, n_cols, figsize=figsize)

        for i in range(cell_pop_summarized_mk.shape[1]):
            r_i = int(i // n_rows)
            c_i = int(i % n_rows)

            ax[c_i, r_i].scatter(
                t_m[sort_order] * self.dataset.time_range + self.dataset.time_min,
                cell_pop_summarized_mk[sort_order, i].clone().detach().cpu(),
                color=cm.tab10(i),
            )

            ax[c_i, r_i].set_title(high_cell_type_str_list[i])

        matplotlib.pyplot.tight_layout()

        return ax

    def plot_sample_compositions_boxplot(self, figsize=(16, 9)):
        figsize = (16, 9)

        t_m = self.dataset.t_m.clone().detach().cpu()
        cell_pop = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        sort_order = torch.argsort(self.dataset.t_m)

        n_cell_types = cell_pop.shape[1]

        n_rows = math.ceil(math.sqrt(n_cell_types))
        n_cols = math.ceil(n_cell_types / n_rows)

        fig, ax = matplotlib.pyplot.subplots(n_rows, n_cols, figsize=figsize)

        for i in range(cell_pop.shape[1]):
            r_i = int(i // n_rows)
            c_i = int(i % n_rows)

            t = t_m[sort_order] * self.dataset.time_range + self.dataset.time_min
            prop = cell_pop[sort_order, i].clone().detach().cpu()
            labels = self.dataset.cell_type_str_list[i]

            df1 = pd.DataFrame({"time": t, "proportion": prop,})

            sns.boxplot(
                x="time", y="proportion", data=df1, ax=ax[c_i, r_i], color=cm.tab10(i)
            )
            ax[c_i, r_i].set_title(self.dataset.cell_type_str_list[i])

        matplotlib.pyplot.tight_layout()

        return ax


def evaluate_model(params: dict, reference_deconvolution: TimeRegularizedDeconvolution):
    # TODO: Update to work with different proportion types

    sim_res = simulate_with_sigmoid_proportions(
        **params["simulation_params"], reference_deconvolution=reference_deconvolution
    )

    simulated_bulk = generate_anndata_from_sim(
        sim_res=sim_res, reference_deconvolution=reference_deconvolution
    )

    simulated_dataset = DeconvolutionDataset(
        bulk_anndata=simulated_bulk, **params["deconvolution_dataset_params"]
    )

    simulated_deconvolution = TimeRegularizedDeconvolution(
        dataset=simulated_dataset, **params["deconvolution_params"]
    )

    simulated_deconvolution.fit_model(**params["fit_params"])

    error = calculate_prediction_error(
        sim_res, simulated_deconvolution, n_intervals=100
    )

    return error


def get_default_evaluation_param(device, dtype, dtype_np):
    default_param = {
        "simulation_params": {"num_samples": 100,},
        "deconvolution_dataset_params": {
            "sc_celltype_col": "Subclustering_reduced",
            "bulk_time_col": "time",
            "dtype_np": dtype_np,
            "dtype": dtype,
            "device": device,
            "feature_selection_method": "common",
        },
        "deconvolution_params": {
            "polynomial_degree": 5,
            "basis_functions": "polynomial",
            "use_betas": True,
            "device": device,
            "dtype": dtype,
        },
        "fit_params": {"n_iters": 1000, "verbose": False, "log_frequency": 1000,},
    }

    return default_param


def evaluate_paramset(
    param_set, sc_anndata, reference_deconvolution, show_progress=True
):
    if show_progress:
        progress_bar = tqdm.tqdm
    else:
        progress_bar = lambda x: x

    results = []
    for evaluation_param in progress_bar(param_set):
        evaluation_param_copy = copy.deepcopy(evaluation_param)
        evaluation_param_copy["deconvolution_dataset_params"]["sc_anndata"] = sc_anndata
        error = evaluate_model(
            evaluation_param_copy, reference_deconvolution=reference_deconvolution
        )
        result = {"param": evaluation_param, "error": error}
        results.append(result)

    return results


def generate_batch(
    dataset: DeconvolutionDataset, device: torch.device, dtype: torch.dtype
):

    return {
        "x_mg": dataset.bulk_raw_gex_mg.clone().detach().to(device).type(dtype),
        "t_m": torch.tensor(dataset.dpi_time_m, device=device, dtype=dtype),
    }
