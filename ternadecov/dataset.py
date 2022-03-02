import sys
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

if "boltons" in sys.modules:
    from boltons.cacheutils import cachedproperty as cached_property
else:
    from functools import cached_property


from typing import List

from ternadecov.stats_helpers import *
from ternadecov.simulator import *
from ternadecov.stats_helpers import *
from ternadecov.hypercluster import *


class SingleCellDataset:
    """A reduced dataset with only single-cell data for use with simulator"""

    def __init__(
        self,
        sc_anndata: anndata.AnnData,
        sc_celltype_col: str,
        dtype_np: np.dtype,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_genes = sc_anndata.X.shape[0]
        self.num_cells = sc_anndata.X.shape[1]
        self.sc_celltype_col = sc_celltype_col
        self.dtype_np = dtype_np
        self.dtype = dtype
        self.device = device
        self.sc_anndata = sc_anndata

    @cached_property
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

    @cached_property
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

    @cached_property
    def num_cell_types(self) -> int:
        return len(self.cell_type_str_list)

    @cached_property
    def cell_type_str_to_index_map(self) -> Dict[str, int]:
        return {
            cell_type_str: index
            for index, cell_type_str in enumerate(self.cell_type_str_list)
        }


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

        self.num_genes = len(selected_genes)

        # Subset the single cell AnnData object
        # self.sc_anndata = sc_anndata[:, sc_anndata.var.index.isin(selected_genes)]
        self.sc_anndata = sc_anndata[:, selected_genes]

        # Subset the bulk object
        # self.bulk_anndata = bulk_anndata[:, sc_anndata.var.index.isin(selected_genes)]
        self.bulk_anndata = bulk_anndata[:, selected_genes]

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

    @cached_property
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

    @cached_property
    def cell_type_str_to_index_map(self) -> Dict[str, int]:
        return {
            cell_type_str: index
            for index, cell_type_str in enumerate(self.cell_type_str_list)
        }

    @cached_property
    def num_cell_types(self) -> int:
        return len(self.cell_type_str_list)

    @cached_property
    def num_samples(self) -> int:
        return self.bulk_anndata.X.shape[0]

    @cached_property
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

    @cached_property
    def bulk_raw_gex_mg(self) -> torch.tensor:
        return torch.tensor(self.bulk_anndata.X, device=self.device, dtype=self.dtype)

    @cached_property
    def t_m(self) -> torch.tensor:
        return torch.tensor(self.dpi_time_m, device=self.device, dtype=self.dtype)

    @cached_property
    def bulk_sample_names(self) -> List[str]:
        return list(self.bulk_anndata.obs.index)
