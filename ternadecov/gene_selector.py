import numpy as np
import torch
import anndata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import math
import copy
import pandas as pd
import scanpy as sc
import typing


class GeneSelector:
    @staticmethod
    def rank_genes():
        pass

    @staticmethod
    def select_features(
        bulk_anndata: anndata.AnnData,
        sc_anndata: anndata.AnnData,
        feature_selection_method: str,
        dispersion_cutoff: int = 5,
    ) -> typing.List[str]:

        selected_genes = ()

        if feature_selection_method == "common":
            selected_genes = list(
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
            selected_genes = list(bulk_anndata.var.index[sel_over])

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

            selected_genes = list(selected_genes_bulk.intersection(selected_genes_sc))
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

            selected_genes = list(
                selected_genes_sc.intersection(set(list(bulk_anndata.var.index)))
            )
        else:
            raise ValueError()

        return selected_genes
