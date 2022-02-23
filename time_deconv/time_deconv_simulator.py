from time_deconv import *

# import matplotlib.pylab as plt
import matplotlib
import matplotlib.pyplot
import torch
import pyro
import anndata
import numpy as np

from time_deconv.stats_helpers import *
import pandas as pd


def generate_anndata_from_sim(sim_res, reference_deconvolution):
    """Generate AnnData object from the simulation results

    :param sim_res: simulation results dictonary
    :param reference_deconvolution: reference deconvolution object
    """

    var_tmp = pd.DataFrame({"gene": reference_deconvolution.dataset.selected_genes})
    var_tmp = var_tmp.set_index("gene")

    return anndata.AnnData(
        X=sim_res["x_ng"].numpy(),
        var=var_tmp,
        obs=pd.DataFrame({"time": sim_res["t_m"]}),
    )


def plot_simulated_proportions(
    sim_res, show_sample_proportions=True, show_trajectories=True
):
    """Plot simulated proportion results

    :param sim_res: simulation results objects
    :param show_sample_proportions: show the generated proportions plot
    :param show_trajectories: show underlying trajectories plot
    """

    true_trajectories = sim_res["trajectory_params"]["trajectories_cm"]

    # Order the time axis
    o = torch.argsort(sim_res["t_m"])

    fig, ax = matplotlib.pyplot.subplots(
        sum((show_sample_proportions, show_trajectories))
    )

    if show_trajectories:
        ax[0].plot(sim_res["t_m"][o], true_trajectories[:, o].T)
        ax[0].set_title("True simulated trajectories")

        ax[0].set_xlabel("Set time")
        ax[0].set_ylabel("Proportions")

    if show_sample_proportions:
        for i in range(sim_res["cell_pop_cm"].shape[0]):
            ax[1].scatter(sim_res["t_m"][o], sim_res["cell_pop_cm"][i, o])
        ax[1].set_title("Simulated proportions in samples")
        ax[1].set_xlabel("Set time")
        ax[1].set_ylabel("Proportions")

    return ax


def simulate_data(
    reference_deconvolution,
    start_time=-5,
    end_time=5,
    num_samples=100,
    lib_size_mean=1e6,
    lib_size_std=2e5,
    use_betas=False,
    dirichlet_alpha=1000,
    trajectory_type="sigmoid",
    trajectory_coef=None,
    trajectory_sample_params = {},
):
    """Simulate bulk data with compositional changes

    :param reference_deconvolution: deconvolution object to be used as reference to get single-cell profiles and coefs
    :param start_time: time start
    :param end_time: time end
    :param num_samples: number of samples to simulate
    :param lib_size_mean: mean library size
    :param lib_size_std: library size standard deviation
    :param use_betas: use beta values from the reference model
    :param dirichlet_alpha: global dirichlet alpha coefficient
    :param trajectory_time: type of trajectory to generate sigmoid or periodic
    :

    :return: dictionary of simulated values and underlying coefficients
    """

    # Number of celltypes are same as in main deconvolution
    num_cell_types = reference_deconvolution.w_hat_gc.shape[1]

    # Sample the times
    t_m = torch.rand((num_samples,)) * (end_time - start_time) + start_time

    if trajectory_type == "sigmoid":
        proportions_sample = sample_sigmoid_proportions(
            num_cell_types, num_samples, t_m, dirichlet_alpha, trajectory_coef, trajectory_sample_params = trajectory_sample_params
        )
    elif trajectory_type == "periodic":
        proportions_sample = sample_periodic_proportions(
            num_cell_types, num_samples, t_m, dirichlet_alpha, trajectory_coef
        )
    elif trajectory_type == "linear":
        proportions_sample = sample_linear_proportions(
            num_cell_types, num_samples, t_m, dirichlet_alpha, trajectory_coef
        )
    else:
        raise Exception("Unkown Trajectory Type")

    cell_pop_cm = proportions_sample["cell_pop_cm"]

    # Get phis and betas from main model
    # phi_g ~ 0.1 - 0.2

    phi_g = pyro.param("log_phi_posterior_loc_g").detach().exp().cpu()
    beta_g = pyro.param("log_beta_posterior_loc_g").detach().exp().cpu()

    # Get celltype profiles from the model
    w_hat_gc = reference_deconvolution.w_hat_gc.detach().cpu()
    if use_betas:
        unnorm_w_hat_gc = w_hat_gc * beta_g[:, None]
    else:
        unnorm_w_hat_gc = w_hat_gc

    # Normalize
    w_gc = unnorm_w_hat_gc / unnorm_w_hat_gc.sum(0)

    # Sample library sizes
    lib_sizes_m = torch.normal(
        mean=torch.full([num_samples], lib_size_mean),
        std=torch.full([num_samples], lib_size_std),
    )

    # Get the NegBinomial means
    # consider: random component on w_gc?
    # b_gc -> gene + celltype specific distortion ( how does inference degrade as this increases )
    # sample b_gc from laplace (mu = 1, beta(scale) = )
    # Gamma(mean = 1, var = 1/rate)
    # rate = concentration = a
    # 1/a = var of gamma distribution
    # sample beta_cg
    mu_mg = lib_sizes_m[:, None] * torch.matmul(cell_pop_cm.T, w_gc.transpose(-1, -2))

    # Sample a full matrix using phis from main model
    x_ng = NegativeBinomialAltParam(mu=mu_mg, phi=phi_g).sample()

    return {
        "cell_pop_cm": cell_pop_cm,
        "t_m": t_m,
        "x_ng": x_ng,
        "trajectory_params": proportions_sample["trajectory_params"],
    }


def sample_trajectories(type, num_cell_types):
    """Generate a random trajectory

    :param type: trajectory type (linear, sigmoid, periodical)
    :param num_cell_types: number of cell types in the trajectory
    """
    if type == "linear":
        return sample_linear_trajectories(num_cell_types)
    elif type == "periodic":
        return sample_periodic_trajectories(num_cell_types)
    elif type == "sigmoid":
        return sample_sigmoid_trajectories(num_cell_types)


######################################################
## Linear
######################################################


def sample_linear_trajectories(num_cell_types, seed=2022):
    torch.manual_seed(seed)
    # y = ax+b
    a = torch.rand(num_cell_types) * 10
    b = torch.rand(num_cell_types) * 20 - 10

    return {"a": a, "b": b}


def sample_linear_proportions(
    num_cell_types, num_samples, t_m, dirichlet_alpha=1e4, trajectory_coef=None
):
    """Generate a sample of linear proportions

    :param num_cell_types: number of cell types to simulate
    :param num_samples: number of samples
    :param t_m: torch tensor of times
    :param dirichlet_alpha: multiplier for normalized dirichlet coefficients

    :return: Dictionary of coefficients
    """

    if trajectory_coef is None:
        trajectory_coef = sample_linear_trajectories(num_cell_types)

    a = trajectory_coef["a"]
    b = trajectory_coef["b"]

    trajectories_cm = torch.zeros(num_cell_types, num_samples)
    for i in range(num_cell_types):
        trajectories_cm[i, :] = torch.Tensor(list(a[i] * x + b[i] for x in t_m))

    # Normalize trajectories_cm
    trajectories_cm = torch.nn.functional.softmax(trajectories_cm, dim=0)

    # For every sample, sample proportions from trajectory
    cell_pop_cm = torch.zeros(num_cell_types, num_samples)
    for j in range(num_samples):
        cell_pop_cm[:, j] = torch.distributions.dirichlet.Dirichlet(
            trajectories_cm[:, j] * dirichlet_alpha
        ).sample()

    # Normalize cell_pop_cm -- Not really needed
    # cell_pop_cm = torch.nn.functional.softmax(cell_pop_cm, dim=0)

    return {
        "trajectory_params": {
            "type": "linear",
            "a": a,
            "b": b,
            "trajectories_cm": trajectories_cm,
        },
        "cell_pop_cm": cell_pop_cm,
    }


######################################################
## Periodic
######################################################


def sample_periodic_trajectories(num_cell_types, seed=2022):
    torch.manual_seed(seed)
    # y = a sin(b*x+c)
    a = torch.rand(num_cell_types) * 10 - 5  # (-5,5)
    b = torch.rand(num_cell_types) * 0.75
    c = torch.rand(num_cell_types) * 5

    return {"a": a, "b": b, "c": c}


def sample_periodic_proportions(
    num_cell_types, num_samples, t_m, dirichlet_alpha=1e4, trajectory_coefficients=None
):
    """Get a sample of periodic cell proportions

    :param num_cell_types: number of cell types to simulate
    :param num_samples: number of samples to simulate
    :param t_m: time points to simulate results for
    :param dirichlet_alpha: global diriechlet concentration
    """
    if trajectory_coefficients is None:
        trajectory_coefficients = sample_periodic_trajectories(
            num_cell_types, num_samples
        )

    a = trajectory_coefficients["a"]
    b = trajectory_coefficients["b"]
    c = trajectory_coefficients["c"]

    trajectories_cm = torch.zeros(num_cell_types, num_samples)
    for i in range(num_cell_types):
        trajectories_cm[i, :] = torch.Tensor(
            list(a[i] * torch.sin(b[i] * x + c[i]) for x in t_m)
        )

    trajectories_cm = torch.nn.functional.softmax(trajectories_cm, dim=0)

    # For every sample, sample proportions from trajectory
    cell_pop_cm = torch.zeros(num_cell_types, num_samples)
    for j in range(num_samples):
        cell_pop_cm[:, j] = torch.distributions.dirichlet.Dirichlet(
            trajectories_cm[:, j] * dirichlet_alpha
        ).sample()

    # Normalize cell_pop_cm -- Not really needed
    cell_pop_cm = torch.nn.functional.softmax(cell_pop_cm, dim=0)

    return {
        "trajectory_params": {
            "type": "periodic",
            "a": a,
            "b": b,
            "c": c,
            "trajectories_cm": trajectories_cm,
        },
        "cell_pop_cm": cell_pop_cm,
    }


######################################################
## Sigmoid
######################################################


def sigmoid(x):
    """Return sigmoid function value"""
    return (1. / (1. + np.exp(-x)))


def sample_sigmoid_trajectories(
        num_cell_types, 
        seed=None, 
        effect_size_min = -1, 
        effect_size_max = 1,           
        shift_min = 10, 
        shift_max = 30):
    
    if seed is not None:
        torch.manual_seed(seed)
        
    effect_size = torch.rand(num_cell_types) * (effect_size_max - effect_size_min) + effect_size_min
    shift = torch.rand(num_cell_types) * (shift_max - shift_min) + shift_min

    return {"effect_size": effect_size, "shift": shift}


def sample_sigmoid_proportions(
    num_cell_types, num_samples, t_m, dirichlet_alpha=1e4, trajectory_coefficients=None, trajectory_sample_params = {},
):
    """Generate a sample of sigmoid proportions

    :param num_cell_types: number of cell types to simulate
    :param num_samples: number of samples
    :param t_m: torch tensor of times
    :param dirichlet_alpha: multiplier for normalized dirichlet coefficients

    :return: Dictionary of coefficients
    """
    if trajectory_coefficients is None:
        trajectory_coefficients = sample_sigmoid_trajectories(
            num_cell_types, num_samples, **trajectory_sample_params
        )

    effect_size = trajectory_coefficients["effect_size"]
    shift = trajectory_coefficients["shift"]

    # Generate trajectories_cm
    trajectories_cm = torch.zeros(num_cell_types, num_samples)
    for i in range(num_cell_types):
        trajectories_cm[i, :] = (
            torch.Tensor(list(sigmoid(effect_size[i] * x + shift[i]) for x in t_m))
        )

    trajectories_cm = torch.nn.functional.softmax(trajectories_cm, dim=0)

    # For every sample, sample proportions from trajectory
    cell_pop_cm = torch.zeros(num_cell_types, num_samples)
    for j in range(num_samples):
        cell_pop_cm[:, j] = torch.distributions.dirichlet.Dirichlet(
            trajectories_cm[:, j] * dirichlet_alpha
        ).sample()

    # Normalize cell_pop_cm -- Not really needed
    # cell_pop_cm = torch.nn.functional.softmax(cell_pop_cm, dim=0)

    return {
        "trajectory_params": {
            "type": "sigmoid",
            "effect_size": effect_size,
            "shift": shift,
            "trajectories_cm": trajectories_cm,
        },
        "cell_pop_cm": cell_pop_cm,
    }


######################################################
## Error Calculation
######################################################


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
        shift = sim_res["trajectory_params"]["shift"]
        effect_size = sim_res["trajectory_params"]["effect_size"]
        num_cell_types = sim_res["trajectory_params"]["effect_size"].shape[0]

        cell_pop_cm = torch.zeros(num_cell_types, num_samples)
        for i in range(num_cell_types):
            cell_pop_cm[i, :] = (
                torch.Tensor(list(sigmoid(effect_size[i] * x + shift[i]) for x in t_m))
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
