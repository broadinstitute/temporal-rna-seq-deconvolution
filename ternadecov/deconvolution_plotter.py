import matplotlib
import matplotlib.pyplot
import torch
import pyro
import math
from matplotlib.pyplot import cm
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple, Dict

from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel
from ternadecov.plotting_functions import generate_posterior_samples

class DeconvolutionPlotter:
    def __init__(self, deconvolution: TimeRegularizedDeconvolutionModel):
        self.deconvolution = deconvolution
        
    def plot_loss(self, filenames=()) -> matplotlib.axes.Axes:
        """Plot of ELBO loss during training from the deconvolution object.
        
        :param self: An instance of self.
        :param filenames: An iterable of filenames to save the plot to.
        
        :return: A matplotlib.axes.Axes object.
        """

        fig, ax = matplotlib.pyplot.subplots()

        ax.plot(self.deconvolution.loss_hist)
        ax.set_title("Losses")
        ax.set_xlabel("iteration")
        ax.set_ylabel("ELBO Loss")

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax
        
    def plot_phi_g_distribution(self, filenames=()) -> matplotlib.axes.Axes:
        """Plot the distribution of $phi_g$ values from the param_store.
        
        :param self: An instance of self
        :param filenames: An iterable of filenames to save the plot to
        
        :return: A matplotlib.axes.Axes object.
        """

        phi_g = pyro.param("log_phi_posterior_loc_g").clone().detach().exp().cpu()

        fig, ax = matplotlib.pyplot.subplots()

        ax.hist(phi_g.numpy(), bins=100)
        ax.set_xlabel("$\phi_g$")
        ax.set_ylabel("Counts")

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax

    def plot_beta_g_distribution(self, filenames=()) -> matplotlib.axes.Axes:
        """Plot distribution of beta_g from the param_store.
        
        :param self: An instance of self
        :param filenames: An iterable of filenames to save the plot to
        
        :return: A matplotlib.axes.Axes object.
        """

        beta_g = pyro.param("log_beta_posterior_loc_g").clone().detach().exp().cpu()

        fig, ax = matplotlib.pyplot.subplots()

        ax.hist(beta_g.numpy(), bins=100)
        ax.set_xlabel("$beta_g$")
        ax.set_ylabel("Counts")

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax    
    
    def plot_sample_compositions_scatter(
        self, 
        figsize=(16, 9), 
        ignore_hypercluster=False, 
        filenames=()
    ):
        """Plot a scatter plot of the sample composition facetted by celltype

        :param self: An instance of self
        :param figsize: tuple of size 2 with figure size information
        :param ignore_hypercluster: ignore hyperclustering and plot individual clusters without summarization
        :param filenames: An iterable of filenames to save the plot to.
        """
        
        if self.deconvolution.dataset.is_hyperclustered:
            if ignore_hypercluster:
                self.__plot_sample_compositions_scatter_default(figsize=figsize)
            else:
                self.__plot_sample_compositions_scatter_hyperclustered(figsize=figsize)
        else:
            if ignore_hypercluster:
                raise ValueError("ignore_hypercluster is not supported for non-hyperclustered objecets")
            self.__plot_sample_compositions_scatter_default(figsize=figsize)
        
        for filename in filenames:
            matplotlib.pyplot.savefig(filename)
            
    def plot_composition_trajectories_via_posterior_sampling(
        self,
        show_iqr: bool = True,
        show_combined: bool = True,
        iqr_alpha: float = 0.2,
        t_begin: float = 0.0,
        t_end: float = 1.0,
        n_bins: int = 1000,
        n_samples_per_bin: int = 2000,
        n_windows: int = 10,
        savgol_polyorder: int = 1,
        figsize: Tuple[float, float] = (3.0, 2.0),
        celltype_summarization: dict = dict(),
        sharey: bool = True,
        lw: float = 1.0,
        cell_type_to_color_dict: Optional[Dict[str, str]] = None,
        filenames=(),
        return_data=False,
        **kwargs,
    ):
        """Plot the composition trajectories by sampling from the posterior.
        
        :param self: An instance of self
        :param show_iqr: Plot the Inter-quantile ranges
        :param show_combined: Show all trajectories on one plot
        :param iqr_alpha: alpha transparency for the IQR ranges
        :param t_begin:
        :param t_end:
        :param n_bins: number of time bins
        :param n_samples_per_bin: number of samples per bin
        :param n_windows: number of windows
        :param savgol_polyorder: smoothing polynomial order
        :param figsize: Figure size
        :param celltype_summarization: celltype summarization dictionary (for plotting only)
        :param sharey: Share the y axis
        :param lw: line width
        :param cell_type_to_color_dict: Cell type to color dictionary
        :param filenames: Filenames to save the plots to
        :param \**kwargs:
            Everything else
            
        """
        
        assert self.deconvolution.trajectory_model_type == 'gp', "plot_composition_trajectories_via_posterior_sampling is only possible for GP deconvolution"

        # obtain posterior samples
        xi_nq, pi_sampled_scn = generate_posterior_samples(
            self.deconvolution,
            t_begin=t_begin,
            t_end=t_end,
            n_bins=n_bins,
            n_samples_per_bin=n_samples_per_bin,
        )
        cell_type_labels = self.deconvolution.dataset.cell_type_str_list

        # optionally, summarize
        if len(celltype_summarization) >= 1:
            pi_sampled_scn = summarize_posterior_samples(
                self.deconvolution, pi_sampled_scn, celltype_summarization
            )
            cell_type_labels = list(celltype_summarization.keys())

        # estimate IQR and smooth
        iqr_lo_cn, iqr_mid_cn, iqr_hi_cn = get_iqr_from_posterior_samples(
            pi_sampled_scn,
            perform_smoothing=True,
            n_windows=n_windows,
            savgol_polyorder=savgol_polyorder,
        )

        # plotting

        # take care of colors
        if cell_type_to_color_dict is None:
            cell_type_to_color_dict = self.deconvolution.dataset.cell_type_to_color_dict

        for cell_type in cell_type_labels:
            assert (
                cell_type in cell_type_to_color_dict
            ), f"Color for cell type {cell_type} is not specified!"
        colors = list(map(cell_type_to_color_dict.get, cell_type_labels))

        n_cell_types = pi_sampled_scn.shape[1]
        xi_n = xi_nq.cpu().numpy()[:, 0]

        if show_combined:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ncols = kwargs["ncols"]
            nrows = int(np.ceil(len(cell_type_labels) / ncols))
            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(figsize[0] * ncols, figsize[1] * nrows),
                sharey=sharey,
            )

        actual_time_n = (
            self.deconvolution.dataset.time_min
            + self.deconvolution.dataset.time_range * xi_n
        )

        for i_cell_type in range(n_cell_types):
            color = colors[i_cell_type]

            if not show_combined:
                ax = axs.flatten()[i_cell_type]

            ax.plot(
                actual_time_n,
                iqr_mid_cn[i_cell_type],
                c=color,
                label=cell_type_labels[i_cell_type],
                lw=lw,
            )

            if iqr_alpha > 0:
                ax.fill_between(
                    actual_time_n,
                    iqr_lo_cn[i_cell_type],
                    iqr_hi_cn[i_cell_type],
                    alpha=iqr_alpha,
                    color=color,
                    edgecolor="none",
                )

            ax.set_xlabel("Time")
            if show_combined:
                ax.set_title("Predicted cell proportions")
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
            else:
                ax.set_title(f"{cell_type_labels[i_cell_type]}")

            ax.set_xlim((np.min(actual_time_n), np.max(actual_time_n)))

        # get rid of extra axes
        if not show_combined:
            for idx in range(i_cell_type + 1, ncols * nrows):
                axs.flatten()[idx].axis("off")

        fig.tight_layout()

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        if return_data:
            return {
                "actual_time_n": actual_time_n,
                "iqr_mid_cn": iqr_mid_cn,
                "cell_type_labels": cell_type_labels,
            }
        
    def plot_gp_composition_trajectories(self, n_samples=500, filenames=()):
        """" Plot per-celltype (deprecated)
        
        :param self: An instance of self
        :param n_samples: Number of samples to draw from GP
        :param filenames: Filenames to save to

        """
        
        assert self.deconvolution.trajectory_model_type == 'gp', "plot_composition_trajectories_via_posterior_sampling is only possible for GP deconvolution"
        
        with torch.no_grad():
            traj = self.deconvolution.population_proportion_model
            xi_new_nq = torch.linspace(
                0.0,
                1.0,
                n_samples,
                device=self.deconvolution.device,
                dtype=self.deconvolution.dtype,
            )[..., None]
            f_new_loc_cn, f_new_var_cn = traj.gp.forward(xi_new_nq, full_cov=False)
            f_new_scale_cn = f_new_var_cn.sqrt()
            f_new_sampled_scn = torch.distributions.Normal(
                f_new_loc_cn, f_new_scale_cn
            ).sample([n_samples])

            pi_new_sampled_scn = torch.softmax(f_new_sampled_scn, dim=1)
            # pi_new_loc_cn = torch.softmax(f_new_loc_cn, dim=0)

            plotrange_kcn = torch.quantile(
                pi_new_sampled_scn,
                torch.Tensor([0.25, 0.5, 0.75]).to(self.deconvolution.device),
                0,
            ).cpu()

        n_celltypes = plotrange_kcn.shape[1]
        nrow = math.ceil(math.sqrt(n_celltypes))
        ncol = math.ceil(math.sqrt(n_celltypes))

        fig, ax = matplotlib.pyplot.subplots(nrow, ncol, figsize=(10, 8))

        # TODO: Add colors, add titles
        for i in range(n_celltypes):
            ax[i // nrow, i % nrow].fill_between(
                xi_new_nq.cpu().numpy()[:, 0],
                plotrange_kcn[0, i, :].numpy().T,
                plotrange_kcn[2, i].numpy().T,
            )
            ax[i // nrow, i % nrow].plot(
                xi_new_nq.cpu().numpy(), plotrange_kcn[1, i].cpu().numpy().T, c="black"
            )

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

    def plot_sample_compositions_boxplot_confidence(
        self,
        n_draws=100,
        verbose=False,
        figsize=(20, 15),
        dpi=80,
        spacing=1,
        filenames=(),
    ):
        # l -- draw index

        assert self.deconvolution.trajectory_model_type == "gp"

        n_samples = self.deconvolution.dataset.num_samples
        n_celltypes = self.deconvolution.dataset.num_cell_types
        cell_pop_lmc = torch.zeros([n_draws, n_samples, n_celltypes])
        sort_order = torch.argsort(self.deconvolution.dataset.t_m)

        # times_sorted = self.deconvolution.dataset.t_m[sort_order]

        fig_nrow = math.ceil(math.sqrt(n_celltypes))
        fig_ncol = math.ceil(math.sqrt(n_celltypes))

        # Generate draws from posterior
        for i in range(n_draws):
            cell_pop_lmc[i, :] = (
                self.deconvolution.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )[None, :]

        # Get quantiles of draw
        plot_quantiles = torch.quantile(cell_pop_lmc, q=torch.linspace(0, 1, 5), dim=0)

        # Generate figure and axis
        fig, ax = matplotlib.pyplot.subplots(
            fig_nrow, fig_ncol, figsize=figsize, dpi=dpi
        )

        # Get plotting positions
        z = self.deconvolution.dataset.t_m[sort_order].cpu()

        extra_spacing = torch.where(
            torch.diff(z, append=z[None, -1]) > 1e-6, spacing, 0
        )
        positions = torch.cumsum(torch.ones(extra_spacing.shape) + extra_spacing, 0)

        for c in range(plot_quantiles.shape[2]):  # celltypes
            if verbose:
                print(f"Processing {self.deconvolution.dataset.cell_type_str_list[c]}")

            # Generate data for this panel
            plot_data = list()
            for m in range(plot_quantiles.shape[1]):  # samples
                m = sort_order[m]
                plot_data.append(
                    {
                        "whislo": plot_quantiles[0, m, c].item(),
                        "q1": plot_quantiles[1, m, c].item(),
                        "med": plot_quantiles[2, m, c].item(),
                        "q3": plot_quantiles[3, m, c].item(),
                        "whishi": plot_quantiles[4, m, c].item(),
                        "label": f"{self.deconvolution.dataset.bulk_sample_names[m]}",
                    }
                )

            # Plot panel
            boxprops = dict(facecolor=cm.tab10(c))
            cur_axis = ax[c // fig_nrow, c % fig_nrow]
            cur_axis.bxp(
                bxpstats=plot_data,
                showfliers=False,
                shownotches=False,
                showmeans=False,
                boxprops=boxprops,
                patch_artist=True,
                positions=positions,
            )
            cur_axis.set_title(self.deconvolution.dataset.cell_type_str_list[c])
            cur_axis.set_xticklabels(
                list(
                    self.deconvolution.dataset.bulk_sample_names[x.item()]
                    for x in sort_order
                ),
                rotation=90,
            )

        fig.tight_layout()

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)
    

    def plot_sample_compositions_boxplot(self, figsize=(16, 9), filenames=()):
        """Plot sample compositions in boxplot form
        
        :param self: An instance of self.
        :param figsize: Figure size
        :param filenames: Filename to save the plots to
        
        """

        if self.deconvolution.trajectory_model_type == "polynomial":
            cell_pop = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        elif self.deconvolution.trajectory_model_type == "gp":
            cell_pop = (
                self.deconvolution.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )

        t_m = self.deconvolution.dataset.t_m.clone().detach().cpu()

        sort_order = torch.argsort(self.deconvolution.dataset.t_m)

        n_cell_types = cell_pop.shape[1]

        n_rows = math.ceil(math.sqrt(n_cell_types))
        n_cols = math.ceil(n_cell_types / n_rows)

        fig, ax = matplotlib.pyplot.subplots(n_rows, n_cols, figsize=figsize)

        for i in range(cell_pop.shape[1]):
            r_i = int(i // n_rows)
            c_i = int(i % n_rows)

            t = (
                t_m[sort_order] * self.deconvolution.dataset.time_range
                + self.deconvolution.dataset.time_min
            )
            prop = cell_pop[sort_order, i].clone().detach().cpu()

            df1 = pd.DataFrame({"time": t, "proportion": prop})

            sns.boxplot(
                x="time", y="proportion", data=df1, ax=ax[c_i, r_i], color=cm.tab10(i)
            )
            ax[c_i, r_i].set_title(self.deconvolution.dataset.cell_type_str_list[i])

        matplotlib.pyplot.tight_layout()

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax



    def plot_composition_trajectories(
        self, 
        show_hypercluster=False, 
        show_sampled_trajectories = False,
        filenames=(), 
        **kwargs
    ):
        """Plot the inferred composition trajectories
        
        :param self: An instance of self
        :param show_hypercluster: Show hyper cluster
        :param show_sampled_trajectories: 
        
        """

        if show_sampled_trajectories:

            if self.deconvolution.dataset.is_hyperclustered and not show_hypercluster:
                raise NotImplementedError

            else:
                # TODO: Fix x axis scale

                n_samples = kwargs["n_samples"]

                with torch.no_grad():
                    traj = self.deconvolution.population_proportion_model
                    xi_new_nq = torch.linspace(
                        0.0,
                        1.0,
                        n_samples,
                        device=self.deconvolution.device,
                        dtype=self.deconvolution.dtype,
                    )[..., None]
                    f_new_loc_cn, f_new_var_cn = traj.gp.forward(
                        xi_new_nq, full_cov=False
                    )
                    f_new_scale_cn = f_new_var_cn.sqrt()
                    f_new_sampled_scn = torch.distributions.Normal(
                        f_new_loc_cn, f_new_scale_cn
                    ).sample([n_samples])
                    pi_new_sampled_scn = torch.softmax(f_new_sampled_scn, dim=1)
                    pi_new_loc_cn = torch.softmax(f_new_loc_cn, dim=0)

                fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8))

                # plot the mean trajectory
                ax.plot(xi_new_nq.cpu().numpy(), pi_new_loc_cn.cpu().numpy().T)

                # plot samples
                prop_cycle = matplotlib.pyplot.rcParams["axes.prop_cycle"]
                colors = prop_cycle.by_key()["color"]
                for i_cell_type in range(pi_new_loc_cn.shape[0]):
                    color = colors[i_cell_type]
                    ax.scatter(
                        x=xi_new_nq.expand((n_samples,) + xi_new_nq.shape)
                        .cpu()
                        .numpy()
                        .flatten(),
                        y=pi_new_sampled_scn[:, i_cell_type, :].cpu().numpy().flatten(),
                        c=color,
                        alpha=0.1,
                        s=0.5,
                    )

        else:
            traj = self.deconvolution.population_proportion_model.get_composition_trajectories(
                self.deconvolution.dataset, n_intervals=100
            )

            if self.deconvolution.dataset.is_hyperclustered and not show_hypercluster:
                fig, ax = matplotlib.pyplot.subplots()
                ax.plot(
                    traj["true_times_z"], traj["summarized_composition_rt"].T,
                )
                ax.set_title("Predicted cell proportions")
                ax.set_xlabel("Time")

                labels = []

                r = traj["toplevel_cell_map"]
                map_r = {r[k]: k for k in r}
                for i in range(len(map_r)):
                    labels.append(map_r[i])
                ax.legend(labels, loc="best", fontsize="small")
            else:
                fig, ax = matplotlib.pyplot.subplots()
                ax.plot(
                    traj["true_times_z"], traj["norm_comp_tc"],
                )
                ax.set_title("Predicted cell proportions")
                ax.set_xlabel("Time")
                ax.legend(
                    self.deconvolution.dataset.cell_type_str_list,
                    loc="best",
                    fontsize="small",
                )

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

    def plot_summarized_cell_compositions(
        self, celltype_summarization, n_intervals=100, filenames=(), **kwargs
    ):
        """Plot the composition trajectories"""

        traj = self.deconvolution.population_proportion_model.get_composition_trajectories(
            self.deconvolution.dataset, n_intervals=n_intervals
        )

        times = traj["true_times_z"]
        composition = traj["norm_comp_tc"]
        celltype_labels = self.deconvolution.dataset.cell_type_str_list

        index_v = torch.zeros((len(celltype_labels),), dtype=torch.int32)
        for i, x in enumerate(celltype_labels):
            for k, t in enumerate(celltype_summarization.keys()):
                if x in celltype_summarization[t]:
                    index_v[i] = k

        # k is the summarized c
        composition_summarized_tk = torch.zeros(
            (traj["norm_comp_tc"].shape[0], len(celltype_summarization))
        )
        composition_summarized_tk.index_add_(dim=1, index=index_v, source=composition)

        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(times, composition_summarized_tk)
        ax.set_title("Predicted cell proportions")
        ax.set_xlabel("Time")
        ax.legend(
            celltype_summarization.keys(), loc="best", fontsize="small",
        )
        ax.set_ylim([0, 1])

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def __plot_sample_compositions_scatter_default(self, figsize, filenames=()):
        """Plot a facetted scatter plot of the individual sample compositions for regular processing

        :param figsize: tuple of size 2 with figure size information
        """
        t_m = self.deconvolution.dataset.t_m.clone().detach().cpu()

        if self.deconvolution.trajectory_model_type == "polynomial":
            cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        elif self.deconvolution.trajectory_model_type == "gp":
            cell_pop_mc = (
                self.deconvolution.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )
        elif self.deconvolution.trajectory_model_type == "nontrajectory":
            raise NotImplementedError(
                "Scatter plotting for non-trajectory not implemented"
            )
        else:
            raise NotImplementedError("Unknown trajectory model type")

        sort_order = torch.argsort(self.deconvolution.dataset.t_m)

        n_cell_types = cell_pop_mc.shape[1]

        n_rows = math.ceil(math.sqrt(n_cell_types))
        n_cols = math.ceil(n_cell_types / n_rows)

        fig, ax = matplotlib.pyplot.subplots(n_rows, n_cols, figsize=figsize)

        for i in range(cell_pop_mc.shape[1]):
            r_i = int(i // n_rows)
            c_i = int(i % n_rows)

            ax[c_i, r_i].scatter(
                t_m[sort_order] * self.deconvolution.dataset.time_range
                + self.deconvolution.dataset.time_min,
                cell_pop_mc[sort_order, i].clone().detach().cpu(),
                color=cm.tab10(i),
            )
            ax[c_i, r_i].set_title(self.deconvolution.dataset.cell_type_str_list[i])

        matplotlib.pyplot.tight_layout()

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax

    def __plot_sample_compositions_scatter_hyperclustered(self, figsize, filenames=()):
        """Plot a facetted scatter plot of the individual sample compositions for hyperclustered processing

        :param figsize: tuple of size 2 with figure size information
        """

        assert self.deconvolution.dataset.is_hyperclustered

        t_m = self.deconvolution.dataset.t_m.clone().detach().cpu()

        if self.deconvolution.trajectory_model_type == "polynomial":
            cell_pop_mc = pyro.param("cell_pop_posterior_loc_mc").clone().detach().cpu()
        elif self.deconvolution.trajectory_model_type == "gp":
            cell_pop_mc = (
                self.deconvolution.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )

        sort_order = torch.argsort(self.deconvolution.dataset.t_m)

        # Summarise cell_pop_mc to the high-level clusters
        n_top_level_clusters = len(
            set(self.deconvolution.dataset.hypercluster_results["cluster_map"].values())
        )
        n_low_level_clusters = len(
            set(self.deconvolution.dataset.hypercluster_results["cluster_map"].keys())
        )
        # k is index for  highlevel clusters
        cell_pop_summarized_mk = torch.zeros(
            (cell_pop_mc.shape[0], n_top_level_clusters)
        )

        # Low level cluster names
        low_cell_type_str_list = self.deconvolution.dataset.cell_type_str_list
        toplevel_cell_map = self.deconvolution.calculated_trajectories[
            "toplevel_cell_map"
        ]
        high_cell_type_str_list = list(toplevel_cell_map.keys())
        low_to_high_clustermap = self.deconvolution.dataset.hypercluster_results[
            "cluster_map"
        ]

        index = torch.zeros((n_low_level_clusters,), dtype=torch.int64)

        for i_llc in range(n_low_level_clusters):
            llc_name = low_cell_type_str_list[i_llc]
            hlc_name = low_to_high_clustermap[llc_name]
            i_hlcc = high_cell_type_str_list.index(hlc_name)
            index[i_llc] = i_hlcc

        cell_pop_summarized_mk.index_add_(1, index, cell_pop_mc)

        n_cell_types = cell_pop_summarized_mk.shape[1]

        n_rows = math.ceil(math.sqrt(n_cell_types))
        n_cols = math.ceil(n_cell_types / n_rows)

        fig, ax = matplotlib.pyplot.subplots(n_rows, n_cols, figsize=figsize)

        for i in range(cell_pop_summarized_mk.shape[1]):
            r_i = int(i // n_rows)
            c_i = int(i % n_rows)

            ax[c_i, r_i].scatter(
                t_m[sort_order] * self.deconvolution.dataset.time_range
                + self.deconvolution.dataset.time_min,
                cell_pop_summarized_mk[sort_order, i].clone().detach().cpu(),
                color=cm.tab10(i),
            )

            ax[c_i, r_i].set_title(high_cell_type_str_list[i])

        matplotlib.pyplot.tight_layout()

        for filename in filenames:
            matplotlib.pyplot.savefig(filename)

        return ax
