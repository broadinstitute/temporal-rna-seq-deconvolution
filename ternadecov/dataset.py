import sys
import numpy as np
import torch
from typing import List, Dict
import anndata

if "boltons" in sys.modules:
    from boltons.cacheutils import cachedproperty as cached_property
else:
    from functools import cached_property

from ternadecov.parametrization import (
    DeconvolutionDatatypeParametrization,
    DeconvolutionDatasetParametrization,
)
from ternadecov.gene_selector import GeneSelector
from ternadecov.hypercluster import hypercluster_anndata


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
        self.num_genes = sc_anndata.n_vars
        self.num_cells = sc_anndata.n_obs
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
        types: DeconvolutionDatatypeParametrization,
        parametrization: DeconvolutionDatasetParametrization,
    ):
        


        self.sc_celltype_col = parametrization.sc_celltype_col
        self.bulk_time_col = parametrization.bulk_time_col

        self.dtype_np = types.dtype_np
        self.dtype = types.dtype
        self.device = types.device

        self.selected_genes = ()
        self.verbose = parametrization.verbose

        # Hypercluster related
        self.is_hyperclustered = parametrization.hypercluster

        selected_genes = GeneSelector.select_features(
            parametrization.bulk_anndata,
            parametrization.sc_anndata,
            feature_selection_method=parametrization.feature_selection_method,
            dispersion_cutoff=parametrization.dispersion_cutoff,
            log_sc_cutoff=parametrization.log_sc_cutoff,
            polynomial_degree=parametrization.polynomial_degree,
        )

        self.num_genes = len(selected_genes)
        if self.verbose:
            print(f"{self.num_genes} genes selected")

        # Subset the single cell AnnData object
        self.sc_anndata = parametrization.sc_anndata[:, selected_genes]

        # Subset the bulk object
        self.bulk_anndata = parametrization.bulk_anndata[:, selected_genes]

        # Perform hyper clustering
        if parametrization.hypercluster:

            self.is_hyperclustered = True
            self.hypercluster_results = hypercluster_anndata(
                anndata_obj=parametrization.sc_anndata,
                original_clustering_name=parametrization.sc_celltype_col,
                **hypercluster_params,
            )

            self.sc_anndata.obs = self.sc_anndata.obs.join(
                self.hypercluster_results["new_clusters"]
            )
            self.sc_celltype_col = "hypercluster"

        # Pre-process time values and save inverse function
        self.dpi_time_original_m = self.bulk_anndata.obs[
            parametrization.bulk_time_col
        ].values.astype(self.dtype_np)
        self.time_min = np.min(self.dpi_time_original_m)
        self.time_range = np.max(self.dpi_time_original_m) - self.time_min
        self.dpi_time_m = (self.dpi_time_original_m - self.time_min) / self.time_range

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

    @property
    def cell_type_str_to_index_map(self) -> Dict[str, int]:
        return {
            cell_type_str: index
            for index, cell_type_str in enumerate(self.cell_type_str_list)
        }

    @property
    def num_cell_types(self) -> int:
        return len(self.cell_type_str_list)

    @property
    def num_samples(self) -> int:
        return self.bulk_anndata.X.shape[0]

    @property
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

    @property
    def bulk_raw_gex_mg(self) -> torch.tensor:
        return torch.tensor(self.bulk_anndata.X, device=self.device, dtype=self.dtype)

    @property
    def t_m(self) -> torch.tensor:
        return torch.tensor(self.dpi_time_m, device=self.device, dtype=self.dtype)

    @property
    def bulk_sample_names(self) -> List[str]:
        return list(self.bulk_anndata.obs.index)
