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
        n_samples_per_bin: int = 1000):
    
    with torch.no_grad():
        traj = deconvolution.population_proportion_model
        xi_new_nq = torch.linspace(
            t_begin,
            t_end,
            n_bins,
            device=deconvolution.device,
            dtype=deconvolution.dtype)[..., None]
        f_new_loc_cn, f_new_var_cn = traj.gp.forward(xi_new_nq, full_cov=False)
        f_new_scale_cn = f_new_var_cn.sqrt()
        f_new_sampled_scn = torch.distributions.Normal(
            f_new_loc_cn, f_new_scale_cn).sample([n_samples_per_bin])
        pi_new_sampled_scn = torch.softmax(f_new_sampled_scn, dim=1)
        
    return xi_new_nq, pi_new_sampled_scn


def get_iqr_from_posterior_samples(
        pi_sampled_scn: torch.Tensor,
        perform_smoothing: bool = False,
        n_windows: int = 10,
        savgol_polyorder: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    assert pi_sampled_scn.ndim == 3
    n_samples_per_bin, n_cell_types, n_bins = pi_sampled_scn.shape
    
    with torch.no_grad():
        iqr_kcn = torch.quantile(
            pi_sampled_scn,
            torch.tensor([0.25, 0.5, 0.75], device=pi_sampled_scn.device, dtype=pi_sampled_scn.dtype),
            dim=0).cpu().numpy()
    
    iqr_lo_cn = iqr_kcn[0]
    iqr_mid_cn = iqr_kcn[1]
    iqr_hi_cn = iqr_kcn[2]
    
    if perform_smoothing:
        window_length = n_bins // n_windows
        assert window_length > 1, \
            "Cannot perform smoothing -- increase n_bins in posterior sampling or decreasing n_windows for smoothing!"
        if window_length % 2 == 0:
            window_length += 1
        iqr_lo_cn = savgol_filter(iqr_lo_cn, window_length, savgol_polyorder)
        iqr_mid_cn = savgol_filter(iqr_mid_cn, window_length, savgol_polyorder)
        iqr_hi_cn = savgol_filter(iqr_hi_cn, window_length, savgol_polyorder)
    
    return iqr_lo_cn, iqr_mid_cn, iqr_hi_cn


def summarize_posterior_samples(
        deconvolution: TimeRegularizedDeconvolutionModel,
        pi_sampled_scn: torch.Tensor,
        celltype_summarization: Dict[str, List[str]]) -> torch.Tensor:
    
    # get base cell type labels
    celltype_labels = deconvolution.dataset.cell_type_str_list
    celltype_labels_set = set(celltype_labels)

    # assert the correctness of cell type summarization manifest
    for k, v in celltype_summarization.items():
        for cell_type in v:
            assert cell_type in celltype_labels_set, f"Issue with summarization manifest: cell type {cell_type} is undefined!"
    
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
    pi_summarized_skn = torch.zeros((n_samples_per_bin, n_summarized_cell_types, n_bins))
    pi_summarized_skn.index_add_(dim=1, index=index_v, source=pi_sampled_scn)
    
    return pi_summarized_skn



def plot_composition_trajectories_via_posterior_sampling(
        self,
        show_iqr: bool = True,
        show_combined: bool = True,
        iqr_alpha: float = 0.2,
        t_begin: float = 0.,
        t_end: float = 1.,
        n_bins: int = 1000,
        n_samples_per_bin: int = 2000,
        n_windows: int = 10,
        savgol_polyorder: int = 1,
        celltype_summarization: dict = dict(),
        figsize: Tuple[float, float] = (3., 2.),
        sharey: bool = True,
        lw: float = 1.,
        filenames=(),
        **kwargs):
    """Plot the composition trajectories"""

    # obtain posterior samples
    xi_nq, pi_sampled_scn = generate_posterior_samples(
        self.deconvolution,
        t_begin=t_begin,
        t_end=t_end,
        n_bins=n_bins,
        n_samples_per_bin=n_samples_per_bin)
    cell_type_labels = self.deconvolution.dataset.cell_type_str_list
    
    # optionally, summarize
    if len(celltype_summarization) >= 1:
        pi_sampled_scn = summarize_posterior_samples(
            self.deconvolution,
            pi_sampled_scn,
            celltype_summarization)
        cell_type_labels = list(celltype_summarization.keys())
    
    # estimate IQR and smooth
    iqr_lo_cn, iqr_mid_cn, iqr_hi_cn = get_iqr_from_posterior_samples(
        pi_sampled_scn,
        perform_smoothing=True,
        n_windows=n_windows,
        savgol_polyorder=savgol_polyorder)

    # plot
    prop_cycle = matplotlib.pyplot.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    
    n_cell_types = pi_sampled_scn.shape[1]
    xi_n = xi_nq.cpu().numpy()[:, 0]

    if show_combined:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ncols = kwargs['ncols']
        nrows = int(np.ceil(len(cell_type_labels) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), sharey=sharey)
    
    for i_cell_type in range(n_cell_types):
        color = colors[i_cell_type]
        
        if not show_combined:
            ax = axs.flatten()[i_cell_type]
        
        actual_time_n = self.deconvolution.dataset.time_min + self.deconvolution.dataset.time_range * xi_n
        ax.plot(
            actual_time_n,
            iqr_mid_cn[i_cell_type],
            c=color,
            label=cell_type_labels[i_cell_type],
            lw=lw)
        
        if iqr_alpha > 0:
            ax.fill_between(
                actual_time_n,
                iqr_lo_cn[i_cell_type],
                iqr_hi_cn[i_cell_type],
                alpha=iqr_alpha,
                color=color,
                edgecolor='none')
            
        ax.set_xlabel("Time")
        if show_combined:
            ax.set_title("Predicted cell proportions")
            ax.legend(
                bbox_to_anchor=(1.04, 1),
                loc='upper left',
                fontsize="small")
        else:
            ax.set_title(f'{cell_type_labels[i_cell_type]}')
        
        ax.set_xlim((np.min(actual_time_n), np.max(actual_time_n)))

    # get rid of extra axes
    if not show_combined:
        for idx in range(i_cell_type + 1, ncols * nrows):
            axs.flatten()[idx].axis('off')

    fig.tight_layout()
    
    for filename in filenames:
        matplotlib.pyplot.savefig(filename)