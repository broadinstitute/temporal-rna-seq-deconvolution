"""Stand-alone plotting fuctions, called from DeconvolutionPlotter"""

from scipy.signal import savgol_filter
from typing import Optional, Tuple, Dict
from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel
import torch
import matplotlib.pylab as plt
import matplotlib
import numpy as np
from typing import List


def generate_posterior_samples(
    deconvolution: TimeRegularizedDeconvolutionModel,
    t_begin: float = 0.0,
    t_end: float = 1.0,
    n_bins: int = 1000,
    n_samples_per_bin: int = 10000,
):
    """Generate samples from the posterior of a gp
    
    :param deconvolution: deconvolution model to get posterior samples from
    :param t_begin: start time
    :param t_end: end time
    :param n_bins: number of bins
    :param n_samples_per_bin: number of samples per bin
    
    :return: 
    """

    with torch.no_grad():
        traj = deconvolution.population_proportion_model
        xi_new_nq = torch.linspace(
            t_begin,
            t_end,
            n_bins,
            device=deconvolution.device,
            dtype=deconvolution.dtype,
        )[..., None]
        f_new_loc_cn, f_new_var_cn = traj.gp.forward(xi_new_nq, full_cov=False)
        f_new_scale_cn = f_new_var_cn.sqrt()
        f_new_sampled_scn = torch.distributions.Normal(
            f_new_loc_cn, f_new_scale_cn
        ).sample([n_samples_per_bin])
        pi_new_sampled_scn = torch.softmax(f_new_sampled_scn, dim=1)

    return xi_new_nq, pi_new_sampled_scn


def get_iqr_from_posterior_samples(
    pi_sampled_scn: torch.Tensor,
    perform_smoothing: bool = False,
    n_windows: int = 10,
    savgol_polyorder: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Calculate IQR range from posterior samples
    
    :param pi_sampled_scn: Sampled tensor
    :param perform_smoothing: Flag for performing smoothing
    :param n_windows: Number of windows to smooth 
    :param savgol_polyorder: Polynomial degree for smoothing
    
    :return: tumple of arrays for (0.25, 0.50, 0.75) quantiles
    """

    assert pi_sampled_scn.ndim == 3
    n_samples_per_bin, n_cell_types, n_bins = pi_sampled_scn.shape

    with torch.no_grad():
        iqr_kcn = (
            torch.quantile(
                pi_sampled_scn,
                torch.tensor(
                    [0.25, 0.5, 0.75],
                    device=pi_sampled_scn.device,
                    dtype=pi_sampled_scn.dtype,
                ),
                dim=0,
            )
            .cpu()
            .numpy()
        )

    iqr_lo_cn = iqr_kcn[0]
    iqr_mid_cn = iqr_kcn[1]
    iqr_hi_cn = iqr_kcn[2]

    if perform_smoothing:
        window_length = n_bins // n_windows
        assert (
            window_length > 1
        ), "Cannot perform smoothing -- increase n_bins in posterior sampling or decreasing n_windows for smoothing!"
        if window_length % 2 == 0:
            window_length += 1
        iqr_lo_cn = savgol_filter(iqr_lo_cn, window_length, savgol_polyorder)
        iqr_mid_cn = savgol_filter(iqr_mid_cn, window_length, savgol_polyorder)
        iqr_hi_cn = savgol_filter(iqr_hi_cn, window_length, savgol_polyorder)

    return iqr_lo_cn, iqr_mid_cn, iqr_hi_cn


def summarize_posterior_samples(
    deconvolution: TimeRegularizedDeconvolutionModel,
    pi_sampled_scn: torch.Tensor,
    celltype_summarization: Dict[str, List[str]],
) -> torch.Tensor:
    """Summarize posterior samples by celltype summarization
    
    :param deconvolution: deconvolution object
    :param pi_sampled_scn: Posterior samples to summarize
    :param celltype_summarization: Celltype summarization dictionary
    
    :return: Tensor of summarized posterior samples
    """

    # get base cell type labels
    celltype_labels = deconvolution.dataset.cell_type_str_list
    celltype_labels_set = set(celltype_labels)

    # assert the correctness of cell type summarization manifest
    for k, v in celltype_summarization.items():
        for cell_type in v:
            assert (
                cell_type in celltype_labels_set
            ), f"Issue with summarization manifest: cell type {cell_type} is undefined!"

    # generate an index
    index_v = torch.zeros((len(celltype_labels),), dtype=torch.int32)
    for i, x in enumerate(celltype_labels):
        for k, t in enumerate(celltype_summarization.keys()):
            if x in celltype_summarization[t]:
                index_v[i] = k

    # k is the summarized c
    assert pi_sampled_scn.ndim == 3
    n_samples_per_bin, n_cell_types, n_bins = pi_sampled_scn.shape
    n_summarized_cell_types = len(celltype_summarization)
    pi_summarized_skn = torch.zeros(
        (n_samples_per_bin, n_summarized_cell_types, n_bins)
    )
    pi_summarized_skn.index_add_(dim=1, index=index_v, source=pi_sampled_scn)

    return pi_summarized_skn
