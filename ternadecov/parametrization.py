import torch
import numpy as np


class DeconvolutionDatatypeParametrization:
    """Parametrization for datatypes of model"""

    def __init__(self, device=None, dtype=None, dtype_np=None):
        self.device = torch.device("cuda:0") if device is None else device
        self.dtype = torch.float32 if dtype is None else dtype
        self.dtype_np = np.float32 if dtype_np is None else dtype_np


class TimeRegularizedDeconvolutionModelParametrization:
    def __init__(self):
        # Prior
        self.log_beta_prior_scale = 1.0
        self.tau_prior_scale = 1.0
        self.log_phi_prior_loc = -5.0
        self.log_phi_prior_scale = 1.0

        # Posterior
        self.init_posterior_global_scale_factor = 0.05

        self.log_beta_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.tau_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.log_phi_posterior_loc = -5.0
        self.log_phi_posterior_scale = 0.1 * self.init_posterior_global_scale_factor


class TimeRegularizedDeconvolutionGPParametrization:
    def __init__(self):
        self.init_rbf_kernel_lengthscale = 0.5
        self.init_rbf_kernel_variance = 0.5
        self.init_whitenoise_kernel_variance = 0.1
        self.gp_cholesky_jitter = 1e-4

    @property
    def num_inducing_points(self):
        return self._num_inducing_points

    @num_inducing_points.setter
    def num_inducing_points(self, value):
        self._num_inducing_points = int(value)


class DeconvolutionDatasetParametrization:
    def __init__(
        self,
        sc_anndata,
        sc_celltype_col,
        bulk_anndata,
        bulk_time_col,
        feature_selection_method="overdispersed_bulk_and_high_sc",
    ):
        self.sc_anndata = sc_anndata
        self.sc_celltype_col = sc_celltype_col
        self.bulk_anndata = bulk_anndata
        self.bulk_time_col = bulk_time_col

        # Other params
        self.feature_selection_method = feature_selection_method
        self.verbose = True

        self.hypercluster = False
        self.hypercluster_min_new_cluster_size = 100
        self.hypercluster_min_cells_recluster = 1000
        self.hypercluster_return_anndata = False
        self.hypercluster_subcluster_resolution = 1
        self.hypercluster_type = "louvain"
        self.hypercluster_do_preproc = True
        self.hypercluster_verbose = True

        self.dispersion_cutoff: int = 5
        self.log_sc_cutoff: int = 2
        self.polynomial_degree: int = 2

    @property
    def hypercluster_params(self):
        """The hyperclustering parameters as a dictionary"""
        return {
            "min_new_cluster_size": self.hypercluster_min_new_cluster_size,
            "min_cells_recluster": self.hypercluster_min_cells_recluster,
            "return_anndata": self.hypercluster_return_anndata,
            "subcluster_resolution": self.hypercluster_subcluster_resolution,
            "type": self.hypercluster_type,
            "do_preproc": self.hypercluster_do_preproc,
            "verbose": self.hypercluster_verbose,
        }
