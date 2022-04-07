from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel
import torch
import pandas as pd

class DeconvolutionWriter():
    def __init__(self, deconvolution: TimeRegularizedDeconvolutionModel):
        self.deconvolution = deconvolution
        
    def write_summarized_cell_compositions(self, celltype_summarization, filename=None, n_intervals=100, returnTable=False):
        """Write summarized composition trajectories to csv file"""
        
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
        composition_summarized_tk = torch.zeros((traj["norm_comp_tc"].shape[0],len(celltype_summarization)))
        composition_summarized_tk.index_add_(dim=1,index=index_v,source=composition)
        
        ret_df = pd.DataFrame(
            composition_summarized_tk.numpy(),
            columns=celltype_summarization.keys()
        )
        ret_df['Time'] = times.numpy()
        long_df = pd.melt(ret_df,('Time',), value_vars=('Blood','Tissue'), var_name='Component', value_name='percent')
        
        if filename is not None:
            long_df.to_csv(filename)
        
        if returnTable:
            return long_df
        
    def write_cell_compositions(self, filename=None, n_intervals=100, returnTable = False):
        """Write cell compositions to csv file"""
        traj = self.deconvolution.population_proportion_model.get_composition_trajectories(
            self.deconvolution.dataset, n_intervals=n_intervals
        )

        times = traj["true_times_z"]
        composition = traj["norm_comp_tc"]
        celltype_labels = self.deconvolution.dataset.cell_type_str_list
        
        ret_df = pd.DataFrame(
            composition.numpy(),
            columns = celltype_labels
        )
        ret_df['Time'] = times.numpy()
        long_df = pd.melt(ret_df, ('Time',), value_vars=celltype_labels, var_name='Component', value_name = 'percent')
        
        if filename is not None:
            long_df.to_csv(filename)
            
        if returnTable:
            return long_df