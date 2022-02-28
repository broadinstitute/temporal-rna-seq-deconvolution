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
from ternadecov.time_deconv import *


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


def calculate_prediction_error(sim_res, pseudo_time_reg_deconv_sim, n_intervals=1000):
    """Calculate the prediction error of a deconvolution on simulated results
    :param sim_res: results of a simulation
    :param pseudo_time_reg_deconv_sim: the deconvolution object to evaluate
    :n_intervals: number of intervals over which to evaluate the results
    """

    # TODO: Move to sim_res
    start_time = -5
    end_time = 5
    step = (end_time - start_time) / n_intervals

    t_m = torch.arange(start_time, end_time, step)
    num_samples = t_m.shape[0]

    ## Get the ground truth
    if sim_res["trajectory_params"]["type"] == "sigmoid":
        magnitude = sim_res["trajectory_params"]["magnitude"]
        shift = sim_res["trajectory_params"]["shift"]
        effect_size = sim_res["trajectory_params"]["effect_size"]
        num_cell_types = sim_res["trajectory_params"]["effect_size"].shape[0]

        cell_pop_cm = torch.zeros(num_cell_types, num_samples)
        for i in range(num_cell_types):
            cell_pop_cm[i, :] = (
                torch.Tensor(list(sigmoid(magnitude[i] * x + shift[i]) for x in t_m))
                * effect_size[i]
            )
        ground_truth_proportions_cm = torch.nn.functional.softmax(cell_pop_cm, dim=0).T
    elif sim_res["trajectory_params"]["type"] == "linear":
        a = sim_res["trajectory_params"]["a"]
        b = sim_res["trajectory_params"]["b"]
        num_cell_types = sim_res["trajectory_params"]["trajectories_cm"].shape[0]

        cell_pop_cm = torch.zeros(num_cell_types, num_samples)
        for i in range(num_cell_types):
            cell_pop_cm[i, :] = torch.Tensor(list(a[i] * x + b[i] for x in t_m))
        ground_truth_proportions_cm = torch.nn.functional.softmax(cell_pop_cm, dim=0).T
    elif sim_res["trajectory_params"]["type"] == "periodic":
        a = sim_res["trajectory_params"]["a"]
        b = sim_res["trajectory_params"]["b"]
        c = sim_res["trajectory_params"]["c"]
        num_cell_types = sim_res["trajectory_params"]["trajectories_cm"].shape[0]

        cell_pop_cm = torch.zeros(num_cell_types, num_samples)
        for i in range(num_cell_types):
            cell_pop_cm[i, :] = torch.Tensor(
                list(a[i] * torch.sin(b[i] * x + c[i]) for x in t_m)
            )
        ground_truth_proportions_cm = torch.nn.functional.softmax(cell_pop_cm, dim=0).T
    else:
        raise Exception(
            f'Unknown trajectory type { sim_res["trajectory_params"]["type"] }'
        )

    # Get the predictions
    pseudo_time_reg_deconv_sim.calculate_composition_trajectories(
        n_intervals=n_intervals
    )
    ret_vals = pseudo_time_reg_deconv_sim.calculated_trajectories

    predicted_composition_cm = ret_vals["norm_comp_t"]

    # Calculate L1 and L2 losses
    L1_error = (
        (ground_truth_proportions_cm - predicted_composition_cm).abs().sum([0, 1])
    )
    L1_error_norm = L1_error / n_intervals
    L2_error = (
        (ground_truth_proportions_cm - predicted_composition_cm)
        .pow(2)
        .sum([0, 1])
        .sqrt()
    )
    L2_error_norm = L2_error / n_intervals

    # Calculate L1 and L2 losses on the trajectory shapes

    # Normalize by cell type summing to 1
    ground_truth_proportions_norm_cm = (
        ground_truth_proportions_cm / ground_truth_proportions_cm.sum(-2)
    )
    predicted_composition_norm_cm = (
        predicted_composition_cm / predicted_composition_cm.sum(-2)
    )

    shape_L1_error = (
        (ground_truth_proportions_norm_cm - predicted_composition_norm_cm)
        .abs()
        .sum([0, 1])
    )

    return {
        "L1_error": L1_error,
        "L1_error_norm": L1_error_norm,
        "L2_error": L2_error,
        "L2_error_norm": L2_error_norm,
        "shape_L1_error": shape_L1_error,
    }
