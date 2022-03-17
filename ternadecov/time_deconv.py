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

from ternadecov.stats_helpers import *
from ternadecov.simulator import *
from ternadecov.stats_helpers import *
from ternadecov.hypercluster import *
from ternadecov.dataset import *
from ternadecov.trajectories import *
from ternadecov.parametrization import *

# Indices:
# - c cell type
# - g genes
# - m samples
# - k deformation polynomial degree


def generate_batch(
    dataset: DeconvolutionDataset, device: torch.device, dtype: torch.dtype
):

    return {
        "x_mg": dataset.bulk_raw_gex_mg.clone().detach().to(device).type(dtype),
        "t_m": torch.tensor(dataset.dpi_time_m, device=device, dtype=dtype),
    }


_TRAJECTORY_MODEL_TYPES = {"polynomial", "gp"}


class TimeRegularizedDeconvolutionModel:
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

        # self.population_proportion_model.guide()

        # get the posterior cell populations from the trajectory module
        cell_pop_mc = self.population_proportion_model.guide(
            xi_mq=t_m[..., None].contiguous()
        )

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

    def plot_composition_trajectories(self, show_hypercluster=False, **kwargs):
        """Plot the composition trajectories"""

        if "show_sampled_trajectories" in kwargs.keys():

            if self.dataset.is_hyperclustered and not show_hypercluster:
                raise NotImplementedError

            else:
                # TODO: Fix x axis scale

                n_samples = kwargs["n_samples"]

                with torch.no_grad():
                    traj = self.population_proportion_model
                    xi_new_nq = torch.linspace(
                        0.0, 1.0, n_samples, device=self.device, dtype=self.dtype
                    )[..., None]
                    f_new_loc_cn, f_new_var_cn = traj.gp.forward(
                        xi_new_nq, full_cov=False
                    )
                    f_new_scale_cn = f_new_var_cn.sqrt()
                    f_new_sampled_scn = torch.distributions.Normal(
                        f_new_loc_cn, f_new_scale_cn
                    ).sample([n_samples])
                    pi_new_sampled_scn = torch.softmax(f_new_sampled_scn, dim=1)
                    pi_new_loc_cn = torch.softmax(f_new_loc_cn, dim=0)

                fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8))

                # plot the mean trajectory
                ax.plot(xi_new_nq.cpu().numpy(), pi_new_loc_cn.cpu().numpy().T)

                # plot samples
                prop_cycle = matplotlib.pyplot.rcParams["axes.prop_cycle"]
                colors = prop_cycle.by_key()["color"]
                for i_cell_type in range(pi_new_loc_cn.shape[0]):
                    color = colors[i_cell_type]
                    ax.scatter(
                        x=xi_new_nq.expand((n_samples,) + xi_new_nq.shape)
                        .cpu()
                        .numpy()
                        .flatten(),
                        y=pi_new_sampled_scn[:, i_cell_type, :].cpu().numpy().flatten(),
                        c=color,
                        alpha=0.1,
                        s=0.5,
                    )

        else:
            traj = self.population_proportion_model.get_composition_trajectories(
                self.dataset, n_intervals=100
            )

            if self.dataset.is_hyperclustered and not show_hypercluster:
                fig, ax = matplotlib.pyplot.subplots()
                ax.plot(
                    traj["true_times_z"], traj["summarized_composition_rt"].T,
                )
                ax.set_title("Predicted cell proportions")
                ax.set_xlabel("Time")

                labels = []

                r = traj["toplevel_cell_map"]
                map_r = {r[k]: k for k in r}
                for i in range(len(map_r)):
                    labels.append(map_r[i])
                ax.legend(labels, loc="best", fontsize="small")
            else:
                fig, ax = matplotlib.pyplot.subplots()
                ax.plot(
                    traj["true_times_z"], traj["norm_comp_tc"],
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

    def write_sample_compositions(self, csv_filename, ignore_hypercluster=False):
        """Write sample composition to csv file"""

        if self.dataset.is_hyperclustered and not ignore_hypercluster:
            raise NotImplementedError
        else:
            self.write_sample_composition_default(csv_filename)

    def sample_composition_default(self):
        """Return the sample composition in a pandas DataFrame"""

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

    def write_sample_composition_default(self, csv_filename):
        """Write sample composition proportions to csv file
        
        :param csv_filename: filename of csv file to write to
        """

        composition_df = self.sample_composition_default()
        composition_df.to_csv(csv_filename)

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

        if self.trajectory_model_type == "polynomial":
            cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        elif self.trajectory_model_type == "gp":
            cell_pop_mc = (
                self.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )

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

        if self.trajectory_model_type == "polynomial":
            cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        elif self.trajectory_model_type == "gp":
            cell_pop_mc = (
                self.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )

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

        # todo: not clear if this is even needed
        if self.trajectory_model_type == "polynomial":
            cell_pop = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        elif self.trajectory_model_type == "gp":
            cell_pop = (
                self.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )

        t_m = self.dataset.t_m.clone().detach().cpu()

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
