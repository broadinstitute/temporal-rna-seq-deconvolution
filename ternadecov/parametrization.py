"""Parametrization objects holding for parametrizing datasets and deconvolution execution"""

import torch
import numpy as np
from typing import Dict, Optional


class DeconvolutionDatatypeParametrization:
    """Parametrization for datatypes of model"""

    def __init__(self, device=None, dtype=None, dtype_np=None):
        """Initializer for DeconvolutionDatatypeParametrization
        
        :param self: Instance of class
        :param device: torch device to use
        :param dtype: torch dtype to use
        :param dtype_np: numpy dtype to use
        """

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.dtype = torch.float32 if dtype is None else dtype
        self.dtype_np = np.float32 if dtype_np is None else dtype_np


class TimeRegularizedDeconvolutionModelParametrization:
    """Parametrizaition for TimeRegularizedDeconvolutionModel object"""

    def __init__(
        self,
        log_beta_prior_scale=1.0,
        tau_prior_scale=1.0,
        log_phi_prior_loc=-5.0,
        log_phi_prior_scale=1.0,
        init_posterior_global_scale_factor=0.05,
        log_beta_posterior_scale_factor=1.0,
        tau_posterior_scale_factor=1.0,
        log_phi_posterior_loc_factor=-5.0,
        log_phi_posterior_scale_factor=0.1,
    ):
        """Initializer for TimeRegularizedDeconvolutionModelParametrization
        
        :param self: Instance of object
        :param log_beta_prior_scale: Gene beta prior scale value (mean is fixed)
        :param tau_prior_scale: Deprecated (?)
        :param log_phi_prior_loc: Gene phi prior location value
        :param log_phi_prior_scale: Gene phi prior scale value
        :param init_posterior_global_scale_factor: Global scale factor for posterior
        :param log_beta_posterior_scale_factor: Scale factor gene beta posterior  (wrt prior)
        :param tau_posterior_scale_factor: Deprecated(?)
        :param log_phi_posterior_loc_factor: Scale factor phi prior location value
        :param log_phi_posterior_scale_factor: Scale factor phi prior scale value
        """

        # Prior
        self.log_beta_prior_scale = log_beta_prior_scale
        self.tau_prior_scale = tau_prior_scale
        self.log_phi_prior_loc = log_phi_prior_loc
        self.log_phi_prior_scale = log_phi_prior_scale

        # Posterior
        self.init_posterior_global_scale_factor = init_posterior_global_scale_factor

        self.log_beta_posterior_scale = (
            log_beta_posterior_scale_factor * self.init_posterior_global_scale_factor
        )
        self.tau_posterior_scale = (
            tau_posterior_scale_factor * self.init_posterior_global_scale_factor
        )
        self.log_phi_posterior_loc = log_phi_posterior_loc_factor
        self.log_phi_posterior_scale = (
            log_phi_posterior_scale_factor * self.init_posterior_global_scale_factor
        )


class TimeRegularizedDeconvolutionGPParametrization:
    """Parametrization specific to GP deconvolution"""

    def __init__(
        self,
        init_rbf_kernel_lengthscale=0.5,
        init_rbf_kernel_variance=0.5,
        init_whitenoise_kernel_variance=0.1,
        gp_cholesky_jitter=1e-4,
    ):
        """Initializer for TimeRegularizedDeconvolutionGPParametrization
        
        :param init_rbf_kernel_lengthscale: RBF kernel lengthscale
        :param init_rbf_kernel_variance: RBF kernel variance
        :param init_whitenoise_kernel_variance: Whitenoise kernel variance
        :param gp_cholesky_jitter: Cholesky Jitter Factor
        """

        self.init_rbf_kernel_lengthscale = init_rbf_kernel_lengthscale
        self.init_rbf_kernel_variance = init_rbf_kernel_variance
        self.init_whitenoise_kernel_variance = init_whitenoise_kernel_variance
        self.gp_cholesky_jitter = gp_cholesky_jitter

    @property
    def num_inducing_points(self):
        """Getter for number of inducing points (deprecated, current GP is full not sparse)
        
        :param self: Instance of object
        )"""
        return self._num_inducing_points

    @num_inducing_points.setter
    def num_inducing_points(self, value):
        """Setter for numver of inducing points (deprecated, current GP is full not sparse)
        
        :param self: Instance of object
        :param value: value to set to
        """
        self._num_inducing_points = int(value)


class DeconvolutionDatasetParametrization:
    def __init__(
        self,
        sc_anndata,
        sc_celltype_col,
        bulk_anndata,
        bulk_time_col,
        feature_selection_method="overdispersed_bulk_and_high_sc",
        cell_type_to_color_dict: Optional[Dict[str, str]] = None,
        verbose=True,
        hypercluster=False,
        hypercluster_min_new_cluster_size=100,
        hypercluster_min_cells_recluster=1000,
        hypercluster_return_anndata=False,
        hypercluster_subcluster_resolution=1,
        hypercluster_type="louvain",
        hypercluster_do_preproc=True,
        hypercluster_verbose=True,
        dispersion_cutoff: int = 5,
        log_sc_cutoff: int = 2,
        polynomial_degree: int = 2,
    ):
        """Parametrizaton of Deconvolution Dataset
        
        :param self: Instance of object
        :param sc_anndata: A single-cell AnnData object
        :param sc_celltype_col: Column in sc_anndata .obs indicating the cell-type
        :param bulk_anndata: A bulk rna-seq AnnData object
        :param bulk_time_col: Column in bulk_anndata .obs indicating the time
        :param feature_selection_method: Method for selecting features ("overdispersed_bulk_and_high_sc", "overdispersed_bulk", "single_cell_od", "common")
        :param cell_type_to_color_dict: Optional dictionary of colors
        :param verbose: Verbose output
        :param hypercluster: Flag perform hypeclustering
        :param hypercluster_min_new_cluster_size: Minimum new cluster size
        :param hypercluster_min_cells_recluster: Reclustering
        :param hypercluster_return_anndata: Return the AnnData object
        :param hypercluster_subcluster_resolution: Subcluster resolution
        :param hypercluster_type: hyperclustering algorithm
        :param hypercluster_do_preproc: Flag for performing pre-processing on the anndata object
        :param hypercluster_verbose: Verbosity of hyperclustering routine
        :param dispersion_cutoff: Dispersion cut off for bulk genes
        :param log_sc_cutoff: log(Expr) value cut off for single-cell data
        :param polynomial_degree: Degree of polynomial to fit for identifying over dispersed genes
        """

        # Check provided columns are valid
        assert sc_celltype_col in list(
            sc_anndata.obs.columns
        ), "sc_celltype_col not found in sc_anndata"
        assert bulk_time_col in list(
            bulk_anndata.obs.columns
        ), "bulk_time_col not found in bulk data"

        __feature_selection_methods = (
            "overdispersed_bulk_and_high_sc",
            "overdispersed_bulk",
            "single_cell_od",
            "common",
        )
        assert (
            feature_selection_method in __feature_selection_methods
        ), "Invalid feature selection method"

        self.sc_anndata = sc_anndata
        self.sc_celltype_col = sc_celltype_col
        self.bulk_anndata = bulk_anndata
        self.bulk_time_col = bulk_time_col
        self.feature_selection_method = feature_selection_method
        self.cell_type_to_color_dict = cell_type_to_color_dict
        self.verbose = verbose
        self.hypercluster = hypercluster
        self.hypercluster_min_new_cluster_size = hypercluster_min_new_cluster_size
        self.hypercluster_min_cells_recluster = hypercluster_min_cells_recluster
        self.hypercluster_return_anndata = hypercluster_return_anndata
        self.hypercluster_subcluster_resolution = hypercluster_subcluster_resolution
        self.hypercluster_type = hypercluster_type
        self.hypercluster_do_preproc = hypercluster_do_preproc
        self.hypercluster_verbose = hypercluster_verbose
        self.dispersion_cutoff: int = dispersion_cutoff
        self.log_sc_cutoff: int = log_sc_cutoff
        self.polynomial_degree: int = polynomial_degree

    @property
    def hypercluster_params(self):
        """The hyperclustering parameters as a dictionary (for backward compatibility of some routines)
        
        :param self: Instance of object
        :return: Dictionary of hyperclustering parameters
        """
        return {
            "min_new_cluster_size": self.hypercluster_min_new_cluster_size,
            "min_cells_recluster": self.hypercluster_min_cells_recluster,
            "return_anndata": self.hypercluster_return_anndata,
            "subcluster_resolution": self.hypercluster_subcluster_resolution,
            "type": self.hypercluster_type,
            "do_preproc": self.hypercluster_do_preproc,
            "verbose": self.hypercluster_verbose,
        }
