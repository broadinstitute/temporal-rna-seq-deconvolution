"""Trajectories of cell proportions"""
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.contrib.gp.kernels as kernels
from torch.distributions import constraints
from pyro.contrib.gp.models import VariationalGP
from pyro.nn.module import PyroParam, pyro_method
from pyro.contrib.gp.parameterized import Parameterized
from abc import abstractmethod
from typing import Dict

from ternadecov.parametrization import TimeRegularizedDeconvolutionGPParametrization
from ternadecov.stats_helpers import legendre_coefficient_mat


class TrajectoryModule:
    """The base class of all trajectory modules."""

    def __init__(self):
        super(TrajectoryModule, self).__init__()

    @abstractmethod
    def model(self, xi_mq: torch.Tensor) -> torch.Tensor:
        """TBW."""
        raise NotImplementedError

    @abstractmethod
    def guide(self, xi_mq: torch.Tensor) -> torch.Tensor:
        """TBW."""
        raise NotImplementedError


class ParameterizedTrajectoryModule(TrajectoryModule, Parameterized):
    pass

class NonTrajectoryModule(TrajectoryModule):
    def __init__(
        self,
        num_cell_types: int,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.device = device
        self.dtype = dtype
        self.num_cell_types = num_cell_types
        self.num_samples = num_samples
        
        self.unnorm_cell_pop_loc_prior_mc = np.zeros((self.num_samples, self.num_cell_types))
        self.unnorm_cell_pop_scale_prior_mc = np.ones((self.num_samples, self.num_cell_types))
        
        self.unnorm_cell_pop_loc_posterior_mc = np.zeros((self.num_samples, self.num_cell_types))
        self.unnorm_cell_pop_scale_posterior_mc = np.ones((self.num_samples, self.num_cell_types))
        
    def model(self, xi_mq: torch.Tensor) -> torch.Tensor:  
        
        unnorm_cell_pop_loc_prior_mc = pyro.param(
            "unnorm_cell_pop_loc_prior_mc",
            torch.tensor(
                    self.unnorm_cell_pop_loc_prior_mc,
                    device = self.device,
                    dtype = self.dtype,
            ),
        )
        
        unnorm_cell_pop_scale_prior_mc = pyro.param(
            "unnorm_cell_pop_scale_prior_mc",
            torch.tensor(
                self.unnorm_cell_pop_scale_prior_mc,
                device = self.device,
                dtype = self.dtype,                
            )
        )
        
        unnorm_cell_pop_mc = pyro.sample(
            "unnorm_cell_pop_mc",
            dist.Normal(
                loc = unnorm_cell_pop_loc_prior_mc,
                scale = unnorm_cell_pop_scale_prior_mc,
            ).to_event(2),
        )

        norm_cell_pop_mc = torch.nn.functional.softmax(
            unnorm_cell_pop_mc, 
            dim=-1
        )
        
        assert norm_cell_pop_mc.shape == (self.num_samples, self.num_cell_types,)
        
        return norm_cell_pop_mc
    
    def guide(self, xi_mq: torch.Tensor) -> torch.Tensor:
        
        unnorm_cell_pop_loc_posterior_mc = pyro.param(
            "unnorm_cell_pop_loc_posterior_mc",
            torch.tensor(
                self.unnorm_cell_pop_loc_posterior_mc,
                device = self.device,
                dtype = self.dtype,
            )
        )
        
        unnorm_cell_pop_scale_posterior_mc = pyro.param(
            "unnorm_cell_pop_scale_posterior_mc",
            torch.tensor(
                self.unnorm_cell_pop_scale_posterior_mc,
                device = self.device,
                dtype = self.dtype,                
            )
        )
        
        unnorm_cell_pop_posterior_mc = pyro.sample(
            "unnorm_cell_pop_mc",
            dist.Normal(
                loc = unnorm_cell_pop_loc_posterior_mc,
                scale = unnorm_cell_pop_scale_posterior_mc,
            ).to_event(2)
        )
        
        norm_cell_pop_mc = torch.nn.functional.softmax(
            unnorm_cell_pop_posterior_mc, 
            dim=-1
        )
        
        return norm_cell_pop_mc
        
        
class BasicTrajectoryModule(TrajectoryModule):
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

        # Prior
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

        # Posterior
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

    def model(self, xi_mq: torch.Tensor) -> torch.Tensor:

        t_m = xi_mq[:, 0]

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

        deformation_mc = None
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
            c_kl = legendre_coefficient_mat(self.polynomial_degree, dtype=self.dtype)[
                1:,
            ].to(
                self.device
            )  # drop constant term
            intermediate_legenre_vals_km = torch.matmul(c_kl, t_lm)
            deformation_mc = torch.matmul(
                unnorm_cell_pop_deform_ck, intermediate_legenre_vals_km
            ).transpose(-1, -2)
        else:
            raise NotImplementedError

        # The normalized underlying trajectories, serve as Dirichlet params
        trajectory_mc = torch.nn.functional.softmax(
            unnorm_cell_pop_base_c[None, :] + deformation_mc, dim=-1
        )

        # Per-sample draw with parametrized alpha
        dirichlet_dist = dist.Dirichlet(
            concentration=trajectory_mc * dirichlet_alpha
        ).to_event(1)

        cell_pop_mc = pyro.sample("cell_pop_mc", dirichlet_dist)
        assert cell_pop_mc.shape == (self.num_samples, self.num_cell_types,)

        return cell_pop_mc

    def guide(self, xi_mq: torch.Tensor) -> torch.Tensor:
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
        
        return cell_pop_mc

    def get_composition_trajectories(self, dataset, n_intervals=1000):
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

        calculated_trajectories = {
            "times_z": times_z.numpy(),
            "true_times_z": true_times_z,
            "norm_comp_tc": norm_comp_tc,  # These are the trajectories on the native clusters
            "summarized_composition_rt": summarized_composition_rt,  # These are the trajectories on the summarized results
            "toplevel_cell_map": toplevel_cell_map,
        }

        return calculated_trajectories


class VGPTrajectoryModule(ParameterizedTrajectoryModule):
    def __init__(
        self,
        xi_mq: torch.Tensor,
        num_cell_types: int,
        init_posterior_global_scale_factor: float,
        device: torch.device,
        dtype: torch.dtype,
        parametrization: TimeRegularizedDeconvolutionGPParametrization = TimeRegularizedDeconvolutionGPParametrization(),
    ):
        """TBW.

        :param xi_mq: covariate tensor with shape (num_sample, covariate_n_dim)

        .. note:: in the current model where the only covairate is time, covariate_n_dim == 1

        .. note:: The Gaussian process is specifying a function in a (num_cell_types)-dimensional
          unconstrained Euclidean space. Applying softmax to this function give us the normalized
          cell populations on the (num_cell_types)-dimensional simplex. We refer to the unnormalized
          function as "f" for brevity in the code.

        """
        super(VGPTrajectoryModule, self).__init__()

        self.xi_mq = xi_mq
        self.num_cell_types = num_cell_types
        self.init_posterior_global_scale_factor = init_posterior_global_scale_factor
        self.device = device
        self.dtype = dtype

        assert xi_mq.ndim == 2
        self.num_samples, self.covariate_n_dim = xi_mq.shape
        # self.num_inducing_points = parametrization.num_inducing_points
        self.init_rbf_kernel_lengthscale = parametrization.init_rbf_kernel_lengthscale
        self.init_rbf_kernel_variance = parametrization.init_rbf_kernel_variance
        self.init_whitenoise_kernel_variance = (
            parametrization.init_whitenoise_kernel_variance
        )
        self.gp_cholesky_jitter = parametrization.gp_cholesky_jitter

        # Prior
        # kernel setup
        kernel_rbf = kernels.RBF(
            input_dim=self.covariate_n_dim,
            variance=torch.tensor(
                self.init_rbf_kernel_variance, device=device, dtype=dtype
            ),
            lengthscale=torch.tensor(
                self.init_rbf_kernel_lengthscale, device=device, dtype=dtype
            ),
        )
        kernel_whitenoise = kernels.WhiteNoise(
            input_dim=self.covariate_n_dim,
            variance=torch.tensor(
                self.init_whitenoise_kernel_variance, device=device, dtype=dtype
            ),
        )
        kernel_full = kernels.Sum(kernel_rbf, kernel_whitenoise)

        # mean output
        self.gp_f_mean_c = PyroParam(
            torch.zeros((self.num_cell_types,), device=device, dtype=dtype)
        )

        def f_mean_function(xi_nq: torch.Tensor):
            """Takes the covariate tensor with shape (batch_size, covariate_n_dim) and returns the function
            mean with shape (num_cell_types, batch_size).

            .. note: the shape of the output of GP is permuted.
            """
            assert xi_nq.ndim == 2
            assert xi_nq.shape[-1] == self.covariate_n_dim
            batch_size = xi_nq.shape[0]
            return self.gp_f_mean_c[..., None].expand([self.num_cell_types, batch_size])

        # instantiate VGP model
        self.gp = VariationalGP(
            X=xi_mq,
            y=None,
            kernel=kernel_full,
            likelihood=None,
            mean_function=f_mean_function,
            latent_shape=torch.Size([self.num_cell_types]),
            whiten=True,
            jitter=self.gp_cholesky_jitter,
        )

        # Posterior
        self.gp_init_f_posterior_loc_mc = torch.zeros(
            (self.num_samples, self.num_cell_types), device=device, dtype=dtype
        )

        self.gp_init_f_posterior_scale_mc = torch.ones(
            (self.num_samples, self.num_cell_types), device=device, dtype=dtype
        )

    @pyro_method
    def model(self, xi_mq: torch.Tensor) -> torch.Tensor:
        self.set_mode("model")

        # assert that covariates have the same shape as what given to the initializer
        assert xi_mq.shape == (self.num_samples, self.covariate_n_dim)

        # sample the inducing points (this happens implicitly in the model() call to gp)
        self.gp.set_data(X=xi_mq, y=None)
        f_loc_cm, f_var_cm = self.gp.model()

        assert f_loc_cm.shape == (self.num_cell_types, self.num_samples)
        assert f_var_cm.shape == (self.num_cell_types, self.num_samples)

        # permute the indices and var -> std
        f_loc_mc = f_loc_cm.permute(-1, -2)
        f_scale_mc = f_var_cm.sqrt().permute(-1, -2)

        with pyro.plate("batch"):
            f_mc = pyro.sample(
                "f_mc",
                pyro.distributions.Normal(loc=f_loc_mc, scale=f_scale_mc).to_event(1),
            )

        # finally, apply a softmax to bring the unnormalized cell population ("f) inside
        # the simplex
        cell_pop_mc = torch.softmax(f_mc, -1)
        assert cell_pop_mc.shape == (self.num_samples, self.num_cell_types)

        return cell_pop_mc

    @pyro_method
    def guide(self, xi_mq: torch.Tensor) -> torch.Tensor:
        self.set_mode("guide")

        # sample the posterior of the inducing points (happens implicitly inside the guide() call of gp)
        self.gp.guide()

        # sample the posterior of the unnormalized cell population ("f)
        f_posterior_loc_mc = pyro.param(
            "f_posterior_loc_mc", self.gp_init_f_posterior_loc_mc
        )

        f_posterior_scale_mc = pyro.param(
            "f_posterior_scale_mc", self.gp_init_f_posterior_scale_mc
        )

        # with pyro.plate("batch"):
        #     f_mc = pyro.sample(
        #         "f_mc", pyro.distributions.Delta(v=f_posterior_loc_mc).to_event(1)
        #     )
        with pyro.plate("batch"):
            f_mc = pyro.sample(
                "f_mc",
                pyro.distributions.Normal(
                    loc=f_posterior_loc_mc, scale=f_posterior_scale_mc
                ).to_event(1),
            )

        # finally, apply a softmax to bring the unnormalized cell population ("f) inside
        # the simplex
        cell_pop_mc = torch.softmax(f_mc, -1)
        assert cell_pop_mc.shape == (self.num_samples, self.num_cell_types)

        return cell_pop_mc

    def get_composition_trajectories(self, dataset, n_intervals=1000) -> Dict:
        """Get the composition trajectories"""
        with torch.no_grad():
            xi_new_nq = torch.linspace(
                0.0, 1.0, 1000, device=self.device, dtype=self.dtype
            )[..., None]
            f_new_loc_cn, f_new_var_cn = self.gp.forward(xi_new_nq, full_cov=False)

        # f_new_scale_cn = f_new_var_cn.clone().detach().cpu().sqrt()
        pi_new_loc_cn = torch.softmax(f_new_loc_cn.clone().detach().cpu(), dim=0)

        times_z = xi_new_nq[:, 0].clone().detach().cpu()
        # TODO: May need adjustment for legendre
        true_times_z = times_z * dataset.time_range + dataset.time_min
        norm_comp_tc = pi_new_loc_cn.permute(-1, -2)
        summarized_composition_rt = None
        toplevel_cell_map = None

        # plt.plot(pi_new_loc_cn.cpu().numpy().T)

        calculated_trajectories = {
            "times_z": times_z.numpy(),
            "true_times_z": true_times_z,
            "norm_comp_tc": norm_comp_tc,  # These are the trajectories on the native clusters
            "summarized_composition_rt": summarized_composition_rt,  # These are the trajectories on the summarized results
            "toplevel_cell_map": toplevel_cell_map,
        }

        return calculated_trajectories
