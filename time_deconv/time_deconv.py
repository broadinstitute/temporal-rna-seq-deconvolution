import numpy as np
import matplotlib
import matplotlib.pyplot
from torch.distributions import constraints
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from typing import List, Dict
from boltons.cacheutils import cachedproperty
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

# Indices:

# - c cell type
# - g genes
# - m samples
# - k deformation polynomial degree


class DeconvolutionDataset:
    """This class represents a bulk and single-cell dataset to be deconvolved in tandem"""

    def __init__(
        self,
        sc_anndata: anndata.AnnData,
        sc_celltype_col: str,
        bulk_anndata: anndata.AnnData,
        bulk_time_col: str,
        dtype_np: np.dtype,
        dtype: torch.dtype,
        device: torch.device,
        feature_selection_method: str = "common",
        hypercluster=False,
        hypercluster_params={
            "min_new_cluster_size": 100,
            "min_cells_recluster": 1000,
            "return_anndata": False,
            "subcluster_resolution": 1,
            "type": "louvain",
            "do_preproc": True,
            "verbose": True,
        },
    ):

        self.sc_celltype_col = sc_celltype_col
        self.bul_time_col = bulk_time_col
        self.dtype_np = dtype_np
        self.dtype = dtype
        self.device = device
        self.selected_genes = ()

        ## Hypercluster related
        self.is_hyperclustered = False

        # Select common genes and subset/order anndata objects
        # TODO: Issue warning if too many genes removed
        selected_genes = self.__select_features(
            bulk_anndata, sc_anndata, feature_selection_method=feature_selection_method
        )
        
        print(f'type of selected_genes: {type(selected_genes)}')

        self.num_genes = len(selected_genes)

        # Subset the single cell AnnData object
        self.sc_anndata = sc_anndata[:, sc_anndata.var.index.isin(selected_genes)]

        # Subset the bulk object
        self.bulk_anndata = bulk_anndata[:, sc_anndata.var.index.isin(selected_genes)]

        # Perform hyper clustering
        if hypercluster:
            self.is_hyperclustered = True
            self.hypercluster_results = hypercluster_anndata(
                anndata_obj=sc_anndata,
                original_clustering_name=sc_celltype_col,
                **hypercluster_params,
            )

            self.sc_anndata.obs = self.sc_anndata.obs.join(
                self.hypercluster_results["new_clusters"]
            )
            self.sc_celltype_col = "hypercluster"

        # Pre-process time values and save inverse function
        self.dpi_time_original_m = self.bulk_anndata.obs[bulk_time_col].values.astype(
            dtype_np
        )
        self.time_min = np.min(self.dpi_time_original_m)
        self.time_range = np.max(self.dpi_time_original_m) - self.time_min
        self.dpi_time_m = (self.dpi_time_original_m - self.time_min) / self.time_range

    def __select_features(
        self, bulk_anndata, sc_anndata, feature_selection_method, dispersion_cutoff=5
    ):

        if feature_selection_method == "common":
            self.selected_genes = list(
                set(bulk_anndata.var.index).intersection(set(sc_anndata.var.index))
            )
        elif feature_selection_method == "overdispersed_bulk":
            x_train = np.log(bulk_anndata.X.mean(0) + 1)  # log_mu_g
            y_train = np.log(bulk_anndata.X.var(0) + 1)  # log_sigma_g

            X_train = x_train[:, np.newaxis]
            degree = 3
            model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)

            # Select Genes
            sel_over = (y_train - y_pred > 0.0) & (y_train > dispersion_cutoff)
            self.selected_genes = list(bulk_anndata.var.index[sel_over])

        elif feature_selection_method == "overdispersed_bulk_and_high_sc":
            # Fit polynomial degree
            polynomial_degree = 2
            sc_cutoff = 2  # log scale

            # Select overdispersed in bulk
            x_train = np.log(bulk_anndata.X.mean(0) + 1)  # log_mu_g
            y_train = np.log(bulk_anndata.X.var(0) + 1)  # log_sigma_g

            X_train = x_train[:, np.newaxis]

            model = make_pipeline(
                PolynomialFeatures(polynomial_degree), Ridge(alpha=1e-3)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)

            # Select Genes
            sel_over_bulk = (y_train - y_pred > 0.0) & (y_train > dispersion_cutoff)
            selected_genes_bulk = set(bulk_anndata.var.index[sel_over_bulk])

            # Select highly-expressed in single-cell

            selected_genes_sc = set(
                sc_anndata.var.index[np.log(sc_anndata.X.sum(0) + 1) > sc_cutoff]
            )

            self.selected_genes = list(
                selected_genes_bulk.intersection(selected_genes_sc)
            )
        elif feature_selection_method == "single_cell_od":
            ann_data_working = sc_anndata.copy()

            sc.pp.filter_cells(ann_data_working, min_genes=200)
            sc.pp.filter_genes(ann_data_working, min_cells=3)
            sc.pp.normalize_total(ann_data_working, target_sum=1e4)
            sc.pp.log1p(ann_data_working)
            sc.pp.highly_variable_genes(
                ann_data_working, min_mean=0.0125, max_mean=3, min_disp=0.5
            )
            selected_genes_sc = set(
                ann_data_working.var.highly_variable.index[
                    ann_data_working.var.highly_variable
                ]
            )

            self.selected_genes = list(
                selected_genes_sc.intersection(set(list(bulk_anndata.var.index)))
            )

        return self.selected_genes

    @property
    def cell_type_str_list(self) -> List[str]:
        # return sorted(list(set(self.sc_anndata.obs[self.sc_celltype_col])))
        # Nan safe version
        return sorted(
            list(
                x
                for x in set(self.sc_anndata.obs[self.sc_celltype_col])
                if str(x) != "nan"
            )
        )

    @cachedproperty
    def cell_type_str_to_index_map(self) -> Dict[str, int]:
        return {
            cell_type_str: index
            for index, cell_type_str in enumerate(self.cell_type_str_list)
        }

    @cachedproperty
    def num_cell_types(self) -> int:
        return len(self.cell_type_str_list)

    @cachedproperty
    def num_samples(self) -> int:
        return self.bulk_anndata.X.shape[0]

    @cachedproperty
    def w_hat_gc(self) -> np.ndarray:
        """Calculate the estimate cell profiles"""
        w_hat_gc = np.zeros((self.num_genes, self.num_cell_types))
        for cell_type_str in self.cell_type_str_list:
            i_cell_type = self.cell_type_str_to_index_map[cell_type_str]
            mask_j = self.sc_anndata.obs[self.sc_celltype_col].values == cell_type_str
            w_hat_gc[:, i_cell_type] = np.sum(self.sc_anndata.X[mask_j, :], axis=-2)
            w_hat_gc[:, i_cell_type] = w_hat_gc[:, i_cell_type] / np.sum(
                w_hat_gc[:, i_cell_type]
            )
        return w_hat_gc

    @cachedproperty
    def bulk_raw_gex_mg(self) -> torch.tensor:
        return torch.tensor(self.bulk_anndata.X, device=self.device, dtype=self.dtype)

    @cachedproperty
    def t_m(self) -> torch.tensor:
        return torch.tensor(self.dpi_time_m, device=self.device, dtype=self.dtype)


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
        self.polynomial_degree = polynomial_degree
        self.basis_functions = basis_functions
        self.use_betas = use_betas

        # hyperparameters
        self.log_beta_prior_scale = 1.0
        self.log_r_prior_scale = 1.0
        self.tau_prior_scale = 1.0
        self.log_phi_prior_loc = -5.0
        self.log_phi_prior_scale = 1.0

        #####################################################
        ## Prior
        #####################################################

        self.unnorm_cell_pop_base_prior_loc_c = np.zeros((self.dataset.num_cell_types,))
        self.unnorm_cell_pop_base_prior_scale_c = np.ones(
            (self.dataset.num_cell_types,)
        )

        # dist of coefficients of population deformation polynomial
        self.unnorm_cell_pop_deform_prior_loc_ck = np.zeros(
            (self.dataset.num_cell_types, self.polynomial_degree)
        )
        self.unnorm_cell_pop_deform_prior_scale_ck = np.ones(
            (self.dataset.num_cell_types, self.polynomial_degree)
        )

        # Per sample celltype proportions
        self.cell_pop_prior_loc_cm = (
            np.ones((self.dataset.num_cell_types, self.dataset.num_samples))
            / self.dataset.num_cell_types
        )

        # Dirichlet_alpha prior
        self.dirichlet_alpha_prior = np.ones((1,)) * 1e5

        #####################################################
        ## Posterior
        #####################################################
        self.init_posterior_global_scale_factor = 0.05

        self.log_beta_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.log_r_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.tau_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.log_phi_posterior_loc = -5.0
        self.log_phi_posterior_scale = 0.1 * self.init_posterior_global_scale_factor

        self.unnorm_cell_pop_base_posterior_loc_c = np.zeros(
            (self.dataset.num_cell_types,)
        )
        self.unnorm_cell_pop_base_posterior_scale_c = (
            self.init_posterior_global_scale_factor
            * np.ones((self.dataset.num_cell_types,))
        )

        self.unnorm_cell_pop_deform_posterior_loc_ck = np.zeros(
            (self.dataset.num_cell_types, self.polynomial_degree)
        )
        self.unnorm_cell_pop_deform_posterior_scale_ck = (
            self.init_posterior_global_scale_factor
            * np.ones((self.dataset.num_cell_types, self.polynomial_degree))
        )

        self.cell_pop_posterior_loc_mc = (
            self.init_posterior_global_scale_factor
            * np.ones((self.dataset.num_samples, self.dataset.num_cell_types))
            / self.dataset.num_cell_types
        )

        self.dirichlet_alpha_posterior = (
            self.init_posterior_global_scale_factor * np.ones((1,))
        )

        # cache useful tensors
        self.w_hat_gc = torch.tensor(self.dataset.w_hat_gc, device=device, dtype=dtype)

    def model(
        self,
        x_mg: torch.Tensor,
        t_m: torch.Tensor,
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

        # sample unnorm_cell_pop_base_c
        unnorm_cell_pop_base_c = pyro.sample(
            "unnorm_cell_pop_base_c",
            dist.Normal(
                loc=torch.tensor(
                    self.unnorm_cell_pop_base_prior_loc_c,
                    device=self.device,
                    dtype=self.dtype,
                ),
                scale=torch.tensor(
                    self.unnorm_cell_pop_base_prior_scale_c,
                    device=self.device,
                    dtype=self.dtype,
                ),
            ).to_event(1),
        )
        assert unnorm_cell_pop_base_c.shape == (self.dataset.num_cell_types,)

        # Deformation scale is a learnable parameter now
        unnorm_cell_pop_deform_prior_scale_ck = pyro.param(
            "unnorm_cell_pop_deform_prior_scale_ck",
            torch.tensor(
                self.unnorm_cell_pop_deform_prior_scale_ck,
                device=self.device,
                dtype=self.dtype,
            ),
            constraint=constraints.positive,
        )
        assert unnorm_cell_pop_deform_prior_scale_ck.shape == (
            self.dataset.num_cell_types,
            self.polynomial_degree,
        )

        unnorm_cell_pop_deform_ck = pyro.sample(
            "unnorm_cell_pop_deform_ck",
            dist.Normal(
                loc=torch.tensor(
                    self.unnorm_cell_pop_deform_prior_loc_ck,
                    device=self.device,
                    dtype=self.dtype,
                ),
                scale=unnorm_cell_pop_deform_prior_scale_ck,
            ).to_event(2),
        )
        assert unnorm_cell_pop_deform_ck.shape == (
            self.dataset.num_cell_types,
            self.polynomial_degree,
        )

        dirichlet_alpha = pyro.param(
            "dirichlet_alpha",
            torch.tensor(
                self.dirichlet_alpha_prior,
                device=self.device,
                dtype=self.dtype,
            ),
            constraint=constraints.positive,
        )
        assert dirichlet_alpha.shape == (1,)

        # calculate useful derived variables
        beta_g = log_beta_g.exp()
        phi_g = log_phi_g.exp()

        # Get normalized w_gc
        if self.use_betas:
            unnorm_w_gc = self.w_hat_gc * beta_g[:, None]
        else:
            unnorm_w_gc = self.w_hat_gc

        w_gc = unnorm_w_gc / unnorm_w_gc.sum(0)

        if self.basis_functions == "polynomial":
            tau_km = torch.pow(
                t_m[None, :],
                torch.arange(1, self.polynomial_degree + 1, device=self.device)[
                    :, None
                ],
            )
            deformation_mc = torch.matmul(unnorm_cell_pop_deform_ck, tau_km).transpose(
                -1, -2
            )
        elif self.basis_functions == "legendre":
            # l -- power of the term of the legrenre polynomial
            t_m_prime = 2 * t_m - 1  # discrete times in (-1,1)
            t_lm = torch.pow(
                t_m_prime[None, :],
                torch.arange(0, self.polynomial_degree + 1, device=self.device)[
                    :, None
                ],
            )
            c_kl = legendre_coefficient_mat(self.polynomial_degree, dtype=dtype)[
                1:,
            ].to(
                device
            )  # drop constant term
            intermediate_legenre_vals_km = torch.matmul(c_kl, t_lm)
            deformation_mc = torch.matmul(
                unnorm_cell_pop_deform_ck, intermediate_legenre_vals_km
            ).transpose(-1, -2)

        # The normalized underlying trajectories, serve as Dirichlet params
        trajectory_mc = torch.nn.functional.softmax(
            unnorm_cell_pop_base_c[None, :] + deformation_mc, dim=-1
        )

        per_sample_draw = True
        if per_sample_draw:
            # dirichlet_alpha = torch.tensor([1e4], device=self.device)
            dirichlet_dist = dist.Dirichlet(
                concentration=trajectory_mc * dirichlet_alpha
            ).to_event(1)

            cell_pop_mc = pyro.sample("cell_pop_mc", dirichlet_dist)
        else:
            cell_pop_mc = trajectory_mc

        assert cell_pop_mc.shape == (
            self.dataset.num_samples,
            self.dataset.num_cell_types,
        )

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

        # variational parameters for unnorm_cell_pop_base_c ("B_c")
        unnorm_cell_pop_base_posterior_loc_c = pyro.param(
            "unnorm_cell_pop_base_posterior_loc_c",
            torch.tensor(
                self.unnorm_cell_pop_base_posterior_loc_c,
                device=self.device,
                dtype=self.dtype,
            ),
        )

        # variational parameters for unnorm_cell_pop_deform_c ("R_c")
        unnorm_cell_pop_deform_posterior_loc_ck = pyro.param(
            "unnorm_cell_pop_deform_posterior_loc_ck",
            torch.tensor(
                self.unnorm_cell_pop_deform_posterior_loc_ck,
                device=self.device,
                dtype=self.dtype,
            ),
        )

        # Cell composition
        #         new_code = False
        #         if new_code:
        #              cell_pop_unconstrained_posterior_loc_mc = pyro.param(
        #                 "cell_pop_unconstrained_posterior_loc_mc",
        #                 torch.tensor(
        #                     self.cell_pop_posterior_loc_mc,
        #                     device=self.device,
        #                     dtype=self.dtype,
        #                 ),
        #                 constraint=constraints.simplex,
        #             )
        #             epsilon = 1e-6
        #             # This is on a simplex
        #             cell_pop_posterior_loc_mc = epsilon / self.cell_pop_posterior_loc_mc.shape[-1] +
        #                 (1-epsilon) * cell_pop_unconstrained_posterior_loc_mc
        #         else:
        cell_pop_posterior_loc_mc = pyro.param(
            "cell_pop_posterior_loc_mc",
            torch.tensor(
                self.cell_pop_posterior_loc_mc,
                device=self.device,
                dtype=self.dtype,
            ),
            constraint=constraints.simplex,
        )

        # posterior sample statements
        log_phi_g = pyro.sample(
            "log_phi_g", dist.Delta(v=log_phi_posterior_loc_g).to_event(1)
        )

        log_beta_g = pyro.sample(
            "log_beta_g", dist.Delta(v=log_beta_posterior_loc_g).to_event(1)
        )

        unnorm_cell_pop_base_c = pyro.sample(
            "unnorm_cell_pop_base_c",
            dist.Delta(v=unnorm_cell_pop_base_posterior_loc_c).to_event(1),
        )

        unnorm_cell_pop_deform_ck = pyro.sample(
            "unnorm_cell_pop_deform_ck",
            dist.Delta(v=unnorm_cell_pop_deform_posterior_loc_ck).to_event(2),
        )

        cell_pop_mc = pyro.sample(
            "cell_pop_mc",
            dist.Delta(v=cell_pop_posterior_loc_mc).to_event(2),
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

    def calculate_composition_trajectories(self, n_intervals=100, return_vals=False):
        """Calculate the composition trajectories"""
        # calculate true times
        if self.basis_functions == "polynomial":
            time_step = 1 / n_intervals
            times_z = torch.arange(0, 1, time_step)
            # Take time to appropriate exponent
            times_zk = torch.pow(
                times_z[:, None],
                torch.arange(
                    1,
                    self.polynomial_degree + 1,
                ),
            )
            # get the trained params
            base_composition_post_c = (
                pyro.param("unnorm_cell_pop_base_posterior_loc_c").detach().cpu()
            )
            delta_composition_post_ck = (
                pyro.param("unnorm_cell_pop_deform_posterior_loc_ck").detach().cpu()
            )
            # Calculate the deltas for each time point
            delta_cz = torch.matmul(
                delta_composition_post_ck, times_zk.transpose(-1, -2)
            )
            # normalize
            norm_comp_tc = (
                torch.nn.functional.softmax(
                    base_composition_post_c[:, None] + delta_cz, dim=0
                )
                .numpy()
                .T
            )
            true_times_z = times_z * self.dataset.time_range + self.dataset.time_min
        elif self.basis_functions == "legendre":
            time_step = 2 / n_intervals
            times_z = torch.arange(-1, 1, time_step)
            # Take time to appropriate exponent
            times_zk = torch.pow(
                times_z[:, None],
                torch.arange(
                    1,
                    self.polynomial_degree + 1,
                ),
            )
            # get the trained params
            base_composition_post_c = (
                pyro.param("unnorm_cell_pop_base_posterior_loc_c").detach().cpu()
            )
            delta_composition_post_ck = (
                pyro.param("unnorm_cell_pop_deform_posterior_loc_ck").detach().cpu()
            )
            # Calculate the deltas for each time point
            delta_cz = torch.matmul(
                delta_composition_post_ck, times_zk.transpose(-1, -2)
            )
            # normalize
            norm_comp_tc = (
                torch.nn.functional.softmax(
                    base_composition_post_c[:, None] + delta_cz, dim=0
                )
                .numpy()
                .T
            )
            true_times_z = (
                (times_z + 1) / 2
            ) * self.dataset.time_range + self.dataset.time_min

        norm_comp_ct_torch = torch.Tensor(norm_comp_tc).T
        summarized_composition_rt = None
        toplevel_cell_map = None
        if self.dataset.is_hyperclustered:
            cluster_map = self.dataset.hypercluster_results["cluster_map"]
            toplevel_cell_map = {
                ct: i for i, ct in enumerate({cluster_map[k] for k in cluster_map})
            }
            summarized_num_cells = len(toplevel_cell_map)
            summarized_composition_rt = torch.zeros(
                (summarized_num_cells, norm_comp_tc.shape[0])
            )

            for c_index in range(0, norm_comp_ct_torch.shape[0] - 1):
                low_cluster_name = self.dataset.cell_type_str_list[c_index]
                top_cluster_name = cluster_map[low_cluster_name]
                summarized_composition_rt[toplevel_cell_map[top_cluster_name]].add_(
                    norm_comp_ct_torch[
                        c_index,
                    ]
                )

        self.calculated_trajectories = {
            "times_z": times_z.numpy(),
            "true_times_z": true_times_z,
            "norm_comp_tc": norm_comp_tc,  # These are the trajectories on the native clusters
            "summarized_composition_rt": summarized_composition_rt,  # These are the trajectories on the summarized results
            "toplevel_cell_map": toplevel_cell_map,
        }

        if return_vals:
            return self.calculated_trajectories

    def get_composition_trajectories(self):
        """Return the composition trajectories"""
        return self.calculated_trajectories

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
                self.calculated_trajectories["true_times_z"],
                self.calculated_trajectories["norm_comp_tc"],
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

            df1 = pd.DataFrame(
                {
                    "time": t,
                    "proportion": prop,
                }
            )

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
        "simulation_params": {
            "num_samples": 100,
        },
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
        "fit_params": {
            "n_iters": 1000,
            "verbose": False,
            "log_frequency": 1000,
        },
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
