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
from ternadecov.time_deconv_simulator import *
from ternadecov.stats_helpers import *
from ternadecov.hypercluster import *
from ternadecov.dataset import *


class BasicTrajectoryModule:
    def __init__(
        self,
        basis_functions: str,
        polynomial_degree: int,
        num_cell_types: int,
        num_samples: int,
        init_posterior_global_scale_factor: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self.polynomial_degree = polynomial_degree
        self.basis_functions = basis_functions
        self.num_cell_types = num_cell_types
        self.num_samples = num_samples
        self.init_posterior_global_scale_factor = init_posterior_global_scale_factor

        #####################################################
        ## Prior
        #####################################################
        self.unnorm_cell_pop_base_prior_loc_c = np.zeros((self.num_cell_types,))
        self.unnorm_cell_pop_base_prior_scale_c = np.ones((self.num_cell_types,))

        # dist of coefficients of population deformation polynomial
        self.unnorm_cell_pop_deform_prior_loc_ck = np.zeros(
            (self.num_cell_types, self.polynomial_degree)
        )
        self.unnorm_cell_pop_deform_prior_scale_ck = np.ones(
            (self.num_cell_types, self.polynomial_degree)
        )

        # Per sample celltype proportions
        self.cell_pop_prior_loc_cm = (
            np.ones((self.num_cell_types, self.num_samples)) / self.num_cell_types
        )

        # Dirichlet_alpha prior
        self.dirichlet_alpha_prior = np.ones((1,)) * 1e5

        #####################################################
        ## Posterior
        #####################################################

        self.unnorm_cell_pop_base_posterior_loc_c = np.zeros((self.num_cell_types,))
        self.unnorm_cell_pop_base_posterior_scale_c = (
            self.init_posterior_global_scale_factor * np.ones((self.num_cell_types,))
        )

        self.unnorm_cell_pop_deform_posterior_loc_ck = np.zeros(
            (self.num_cell_types, self.polynomial_degree)
        )
        self.unnorm_cell_pop_deform_posterior_scale_ck = (
            self.init_posterior_global_scale_factor
            * np.ones((self.num_cell_types, self.polynomial_degree))
        )

        self.cell_pop_posterior_loc_mc = (
            self.init_posterior_global_scale_factor
            * np.ones((self.num_samples, self.num_cell_types))
            / self.num_cell_types
        )

    def model(self, t_m: torch.Tensor):

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
        assert unnorm_cell_pop_base_c.shape == (self.num_cell_types,)

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
            self.num_cell_types,
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
            self.num_cell_types,
            self.polynomial_degree,
        )

        dirichlet_alpha = pyro.param(
            "dirichlet_alpha",
            torch.tensor(
                self.dirichlet_alpha_prior, device=self.device, dtype=self.dtype,
            ),
            constraint=constraints.positive,
        )
        assert dirichlet_alpha.shape == (1,)

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

        assert cell_pop_mc.shape == (self.num_samples, self.num_cell_types,)

        return cell_pop_mc

    def guide(self):

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
                self.cell_pop_posterior_loc_mc, device=self.device, dtype=self.dtype,
            ),
            constraint=constraints.simplex,
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
            "cell_pop_mc", dist.Delta(v=cell_pop_posterior_loc_mc).to_event(2),
        )

    def calculate_composition_trajectories(
        self, dataset, n_intervals=100, return_vals=False
    ):
        """Calculate the composition trajectories"""
        # calculate true times
        if self.basis_functions == "polynomial":
            time_step = 1 / n_intervals
            times_z = torch.arange(0, 1, time_step)
            # Take time to appropriate exponent
            times_zk = torch.pow(
                times_z[:, None], torch.arange(1, self.polynomial_degree + 1,),
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
            true_times_z = times_z * dataset.time_range + dataset.time_min
        elif self.basis_functions == "legendre":
            time_step = 2 / n_intervals
            times_z = torch.arange(-1, 1, time_step)
            # Take time to appropriate exponent
            times_zk = torch.pow(
                times_z[:, None], torch.arange(1, self.polynomial_degree + 1,),
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
            true_times_z = ((times_z + 1) / 2) * dataset.time_range + dataset.time_min

        norm_comp_ct_torch = torch.Tensor(norm_comp_tc).T
        summarized_composition_rt = None
        toplevel_cell_map = None
        if dataset.is_hyperclustered:
            cluster_map = dataset.hypercluster_results["cluster_map"]
            toplevel_cell_map = {
                ct: i for i, ct in enumerate({cluster_map[k] for k in cluster_map})
            }
            summarized_num_cells = len(toplevel_cell_map)
            summarized_composition_rt = torch.zeros(
                (summarized_num_cells, norm_comp_tc.shape[0])
            )

            for c_index in range(0, norm_comp_ct_torch.shape[0] - 1):
                low_cluster_name = dataset.cell_type_str_list[c_index]
                top_cluster_name = cluster_map[low_cluster_name]
                summarized_composition_rt[toplevel_cell_map[top_cluster_name]].add_(
                    norm_comp_ct_torch[c_index,]
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
