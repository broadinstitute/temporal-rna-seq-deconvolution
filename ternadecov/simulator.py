import matplotlib
import matplotlib.pyplot
import torch
import pyro
import anndata
import numpy as np
import pandas as pd

from ternadecov.stats_helpers import NegativeBinomialAltParam


def generate_anndata_from_sim(sim_res, reference_dataset):
    """Generate AnnData object from the simulation results

    :param sim_res: simulation results dictonary
    :param reference_deconvolution: reference deconvolution object
    """

    var_tmp = pd.DataFrame({"gene": list(reference_dataset.sc_anndata.var.index)})
    var_tmp = var_tmp.set_index("gene")

    return anndata.AnnData(
        X=sim_res["x_ng"].numpy(),
        var=var_tmp,
        obs=pd.DataFrame({"time": sim_res["t_m"]}),
    )


def plot_simulated_proportions(
    sim_res,
    dataset,
    show_sample_proportions=True,
    show_trajectories=True,
    figsize=(20, 10),
):
    """Plot simulated proportion results

    :param sim_res: simulation results objects
    :dataset: 
    :param show_sample_proportions: show the generated proportions plot
    :param show_trajectories: show underlying trajectories plot
    """

    true_trajectories = sim_res["trajectory_params"]["trajectories_cm"]

    # Order the time axis
    o = torch.argsort(sim_res["t_m"])

    fig, ax = matplotlib.pyplot.subplots(
        sum((show_sample_proportions, show_trajectories)), figsize=figsize
    )

    if show_trajectories:
        if sim_res["trajectory_params"]["type"] == "linear":
            n_samples = 1000
            t_m = (
                np.linspace(0.0, 1.0, n_samples) * dataset.time_range + dataset.time_min
            )
            trajectories_cm = torch.zeros(dataset.num_cell_types, n_samples)

            a = sim_res["trajectory_params"]["a"]
            b = sim_res["trajectory_params"]["b"]

            for i in range(dataset.num_cell_types):
                trajectories_cm[i, :] = torch.Tensor(list(a[i] * x + b[i] for x in t_m))
            # Normalize trajectories_cm
            trajectories_cm = torch.nn.functional.softmax(trajectories_cm, dim=0)

            ax[0].plot(
                t_m, trajectories_cm.T,
            )
            ax[0].legend(dataset.cell_type_str_list)
            ax[0].set_title("True simulated trajectories")
            ax[0].set_xlabel("Set time")
            ax[0].set_ylabel("Proportions")

        else:
            raise NotImplementedError

    if show_sample_proportions:
        for i in range(sim_res["cell_pop_cm"].shape[0]):
            ax[1].scatter(sim_res["t_m"][o], sim_res["cell_pop_cm"][i, o])
        ax[1].set_title("Simulated proportions in samples")
        ax[1].set_xlabel("Set time")
        ax[1].set_ylabel("Proportions")

    return ax


def simulate_data(
    w_hat_gc,
    start_time=-5,
    end_time=5,
    num_samples=100,
    lib_size_mean=1e6,
    lib_size_std=2e5,
    use_betas=False,
    dirichlet_alpha=1000,
    trajectory_type="sigmoid",
    trajectory_coef=None,
    phi_mean=0.15,
    phi_std=0.05,
    beta_mean=1.0,
    beta_std=0.1,
    trajectory_sample_params={},
    seed=None,
):
    """Simulate bulk data with compositional changes

    :param w_hat_gc: reference matrix
    :param start_time: time start
    :param end_time: time end
    :param num_samples: number of samples to simulate
    :param lib_size_mean: mean library size
    :param lib_size_std: library size standard deviation
    :param use_betas: use beta values from the reference model
    :param dirichlet_alpha: global dirichlet alpha coefficient
    :param trajectory_type: type of trajectory ('sigmoid','linear','periodic')
    :param trajectory_coef: predefined trajectory coefficients, if not provided they are sampled
    :param phi_mean: $\phi_{mean}$ value
    :param phi_std: $\phi_{std}$ values
    :param beta_mean: $\beta_{mean}$ values
    :param beta_std: $\beta_{std}$ values
    :param trajectory_sample_params: Dictionary of trajectory sample parameters
    :param seed: seed for trajectory sampling (optional)

    :return: dictionary of simulated values and underlying coefficients
    """

    num_genes = w_hat_gc.shape[0]
    num_cell_types = w_hat_gc.shape[1]

    # Get equidistant time samples
    t_m = torch.linspace(0.0, 1.0, num_samples) * (end_time - start_time) + start_time

    if trajectory_type == "sigmoid":
        proportions_sample = sample_sigmoid_proportions(
            num_cell_types=num_cell_types,
            num_samples=num_samples,
            t_m=t_m,
            dirichlet_alpha=dirichlet_alpha,
            trajectory_coefficients=trajectory_coef,
            trajectory_sample_params=trajectory_sample_params,
            seed=seed,
        )
    elif trajectory_type == "periodic":
        proportions_sample = sample_periodic_proportions(
            num_cell_types=num_cell_types,
            num_samples=num_samples,
            t_m=t_m,
            dirichlet_alpha=dirichlet_alpha,
            trajectory_coefficients=trajectory_coef,
            trajectory_sample_params=trajectory_sample_params,
            seed=seed,
        )
    elif trajectory_type == "linear":
        proportions_sample = sample_linear_proportions(
            num_cell_types,
            num_samples,
            t_m,
            dirichlet_alpha,
            trajectory_coef,
            trajectory_sample_params=trajectory_sample_params,
            seed=seed,
        )
    else:
        raise Exception("Unkown Trajectory Type")

    cell_pop_cm = proportions_sample["cell_pop_cm"]
    phi_g = (
        torch.distributions.normal.Normal(
            loc=torch.full((num_genes,), phi_mean),
            scale=torch.full((num_genes,), phi_std),
        )
        .sample()
        .abs()
    )
    beta_g = (
        torch.distributions.normal.Normal(
            loc=torch.full((num_genes,), beta_mean),
            scale=torch.full((num_genes,), beta_std),
        )
        .sample()
        .abs()
    )

    # Get celltype profiles from the model
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
# Linear
######################################################


def sample_linear_trajectories(
    num_cell_types, seed=None, a_min=0, a_max=10, b_min=-10, b_max=10
):
    if seed is not None:
        torch.manual_seed(seed)

    # y = ax+b
    a = torch.rand(num_cell_types) * (a_max - a_min) + a_min
    b = torch.rand(num_cell_types) * (b_max - b_min) + b_min

    return {"a": a, "b": b}


def sample_linear_proportions(
    num_cell_types,
    num_samples,
    t_m,
    dirichlet_alpha=1e4,
    trajectory_coef=None,
    trajectory_sample_params=None,
    seed=None,
):
    """Generate a sample of linear proportions

    :param num_cell_types: number of cell types to simulate
    :param num_samples: number of samples
    :param t_m: torch tensor of times
    :param dirichlet_alpha: multiplier for normalized dirichlet coefficients

    :return: Dictionary of coefficients
    """

    if trajectory_coef is None:
        trajectory_coef = sample_linear_trajectories(
            num_cell_types, seed=seed, **trajectory_sample_params
        )

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
# Periodic
######################################################


def sample_periodic_trajectories(
    num_cell_types, seed=None, a_min=-5, a_max=5, b_min=0, b_max=0.75, c_min=0, c_max=5
):
    """Get a sample of coefficients for periodic trajectories"""

    if seed is not None:
        torch.manual_seed(seed)

    # y = a sin(b*x+c)
    a = torch.rand(num_cell_types) * (a_max - a_min) + a_min
    b = torch.rand(num_cell_types) * (b_max - b_min) + b_min
    c = torch.rand(num_cell_types) * (c_max - c_min) + c_min

    return {"a": a, "b": b, "c": c}


def sample_periodic_proportions(
    num_cell_types,
    num_samples,
    t_m,
    dirichlet_alpha=1e4,
    trajectory_coefficients=None,
    trajectory_sample_params=None,
    seed=None,
):
    """Get a sample of periodic cell proportions

    :param num_cell_types: number of cell types to simulate
    :param num_samples: number of samples to simulate
    :param t_m: time points to simulate results for
    :param dirichlet_alpha: global diriechlet concentration
    """
    if trajectory_coefficients is None:
        trajectory_coefficients = sample_periodic_trajectories(
            num_cell_types=num_cell_types, seed=seed, **trajectory_sample_params
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
# Sigmoid
######################################################


def sigmoid(x):
    """Return sigmoid function value"""
    return 1.0 / (1.0 + np.exp(-x))


def sample_sigmoid_trajectories(
    num_cell_types,
    seed=None,
    effect_size_min=-1,
    effect_size_max=1,
    shift_min=-2,
    shift_max=2,
):
    """Return sigmoid trajectory param dictionary"""

    if seed is not None:
        torch.manual_seed(seed)

    effect_size = (
        torch.rand(num_cell_types) * (effect_size_max - effect_size_min)
        + effect_size_min
    )
    shift = torch.rand(num_cell_types) * (shift_max - shift_min) + shift_min

    return {"effect_size": effect_size, "shift": shift}


def sample_sigmoid_proportions(
    num_cell_types,
    num_samples,
    t_m,
    dirichlet_alpha=1e4,
    trajectory_coefficients=None,
    trajectory_sample_params={},
    seed=None,
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
            num_cell_types=num_cell_types, seed=seed, **trajectory_sample_params,
        )

    effect_size = trajectory_coefficients["effect_size"]
    shift = trajectory_coefficients["shift"]

    # Generate trajectories_cm
    trajectories_cm = torch.zeros(num_cell_types, num_samples)
    for i in range(num_cell_types):
        trajectories_cm[i, :] = torch.Tensor(
            list(sigmoid(effect_size[i] * x + shift[i]) for x in t_m)
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
# Error Calculation
######################################################
def calculate_sample_prediction_error(sim_res, pseudo_time_reg_deconv_sim):
    # Ground Truth
    ground_truth_cell_pop_cm = sim_res["cell_pop_cm"]
    estimated_cell_pop_cm = (
        pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu().T
    )

    l1_error = (ground_truth_cell_pop_cm - estimated_cell_pop_cm).abs().sum([0, 1])
    l1_error_norm = l1_error / estimated_cell_pop_cm.shape[-1]

    return {"l1_error": l1_error, "l1_error_norm": l1_error_norm}


def calculate_trajectory_prediction_error(
    sim_res, pseudo_time_reg_deconv_sim, n_intervals=1000
):
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

    # Get the ground truth
    if sim_res["trajectory_params"]["type"] == "sigmoid":
        shift = sim_res["trajectory_params"]["shift"]
        effect_size = sim_res["trajectory_params"]["effect_size"]
        num_cell_types = sim_res["trajectory_params"]["effect_size"].shape[0]

        cell_pop_cm = torch.zeros(num_cell_types, num_samples)
        for i in range(num_cell_types):
            cell_pop_cm[i, :] = torch.Tensor(
                list(sigmoid(effect_size[i] * x + shift[i]) for x in t_m)
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
    traj = pseudo_time_reg_deconv_sim.population_proportion_model.get_composition_trajectories(
        dataset=pseudo_time_reg_deconv_sim.dataset, n_intervals=n_intervals
    )
    ret_vals = traj

    predicted_composition_cm = ret_vals["norm_comp_tc"]

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
