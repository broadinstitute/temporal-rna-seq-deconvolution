"""Deconvolution writter for writting output tables"""


from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel
from ternadecov.utils import melt_tensor_to_pandas
import torch
import pandas as pd
from typing import Dict, List


class DeconvolutionWriter:
    def __init__(self, deconvolution: TimeRegularizedDeconvolutionModel):
        self.deconvolution = deconvolution

    def write_summarized_cell_compositions(
        self,
        celltype_summarization: Dict[str, List[str]],
        filename=None,
        n_intervals=100,
        return_table=False,
    ):
        """Write summarized composition trajectories to csv file
        
        :param celltype_summarization: dictionary of lists with the cell summarization to be performed
        :param filename: name of csv file to save table to
        :param n_intervals: number of points to evaluate the trajectories at
        :param return_table: flag to return the resulting data as a pandas DataFrame
        """

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

        ret_df = pd.DataFrame(
            composition_summarized_tk.numpy(), columns=celltype_summarization.keys()
        )
        ret_df["Time"] = times.numpy()
        long_df = pd.melt(
            ret_df,
            ("Time",),
            value_vars=("Blood", "Tissue"),
            var_name="Component",
            value_name="percent",
        )

        if filename is not None:
            long_df.to_csv(filename)

        if returnTable:
            return long_df

    def write_cell_compositions(
        self, filename=None, n_intervals=100, return_table=False
    ):
        """Write cell composition trajectories to csv file
        
        :param filename: name of file to csv data to
        :param n_intervals: number of intervals to evaluate the trajectories at:
        :param return_table: optionally return the calculated table as a pandas df
        
        """
        traj = self.deconvolution.population_proportion_model.get_composition_trajectories(
            self.deconvolution.dataset, n_intervals=n_intervals
        )

        times = traj["true_times_z"]
        composition = traj["norm_comp_tc"]
        celltype_labels = self.deconvolution.dataset.cell_type_str_list

        ret_df = pd.DataFrame(composition.numpy(), columns=celltype_labels)
        ret_df["Time"] = times.numpy()
        long_df = pd.melt(
            ret_df,
            ("Time",),
            value_vars=celltype_labels,
            var_name="Component",
            value_name="percent",
        )

        if filename is not None:
            long_df.to_csv(filename)

        if return_table:
            return long_df

    def write_trajectory_coefficients(self, filename, returnTable=True):
        raise NotImplementedError()

    def write_sample_draws_quantiles(
        self, filename=None, n_draws=100, return_table=True
    ):
        assert self.deconvolution.trajectory_model_type == "gp"

        n_samples = self.deconvolution.dataset.num_samples
        n_celltypes = self.deconvolution.dataset.num_cell_types
        cell_pop_lmc = torch.zeros([n_draws, n_samples, n_celltypes])
        # sort_order = torch.argsort(self.deconvolution.dataset.t_m)
        sample_names = self.deconvolution.dataset.bulk_sample_names
        celltype_labels = self.deconvolution.dataset.cell_type_str_list

        # Generate draws from posterior
        for i in range(n_draws):
            cell_pop_lmc[i, :] = (
                self.deconvolution.population_proportion_model.guide(torch.Tensor([]))
                .clone()
                .detach()
                .cpu()
            )[None, :]

        # Get quantiles of draw
        plot_quantiles_qmc = torch.quantile(
            cell_pop_lmc, q=torch.linspace(0, 1, 5), dim=0
        )

        quantile_labels = list(str(x) for x in range(5))

        df = melt_tensor_to_pandas(
            plot_quantiles_qmc,
            ("quantile", "sample", "celltype"),
            quantile_labels,
            sample_names,
            celltype_labels,
        )

        if filename is not None:
            df.to_csv(filename)

        if return_table:
            return df
