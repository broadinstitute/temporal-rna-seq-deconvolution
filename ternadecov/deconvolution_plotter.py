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


class DeconvolutionPlotter:
    
    def __init__(self, deconvolution):
        self.deconvolution = deconvolution
        
    
        
            
    def plot_sample_compositions_boxplot_confidence(self, n_draws=100, verbose=False, figsize=(20,15), dpi=80,
                                                           spacing = 1):
        # l -- draw index
        
        assert self.deconvolution.trajectory_model_type == "gp"
  
        n_samples = self.deconvolution.dataset.num_samples
        n_celltypes = self.deconvolution.dataset.num_cell_types
        cell_pop_lmc = torch.zeros([n_draws, n_samples, n_celltypes])
        sort_order = torch.argsort(self.deconvolution.dataset.t_m)
        
        #times_sorted = self.deconvolution.dataset.t_m[sort_order]
        
        fig_nrow = math.floor(math.sqrt(n_celltypes))
        fig_ncol = math.ceil(math.sqrt(n_celltypes))

        # Generate draws from posterior
        for i in range(n_draws):
            cell_pop_lmc[i,:] = (
                        self.deconvolution.population_proportion_model.guide(torch.Tensor([]))
                        .clone()
                        .detach()
                        .cpu()
                    )[None,:]

        # Get quantiles of draw
        plot_quantiles = torch.quantile(cell_pop_lmc, q=torch.linspace(0,1,5), dim=0)

        # Generate figure and axis
        fig, ax = matplotlib.pyplot.subplots(fig_nrow, fig_ncol, figsize=figsize, dpi=dpi)
        
        # Get plotting positions
        z = self.deconvolution.dataset.t_m[sort_order]

        extra_spacing = torch.where(torch.diff(z, append=z[None,-1]) > 1e-6, spacing, 0)
        positions =  torch.cumsum(torch.ones(extra_spacing.shape) + extra_spacing, 0)

        for c in range(plot_quantiles.shape[2]): # celltypes
            if verbose:
                print(f"Processing {self.deconvolution.dataset.cell_type_str_list[c]}")

            # Generate data for this panel
            plot_data = list()
            for m in range(plot_quantiles.shape[1]): # samples
                m = sort_order[m]
                plot_data.append({
                    "whislo": plot_quantiles[0, m, c].item(),
                    "q1": plot_quantiles[1, m, c].item(),
                    "med": plot_quantiles[2, m, c].item(),
                    "q3": plot_quantiles[3, m, c].item(),
                    "whishi": plot_quantiles[4, m, c].item(),
                    "label": f"{self.deconvolution.dataset.bulk_sample_names[m]}",
                })

            # Plot panel
            boxprops = dict(facecolor=cm.tab10(c))
            cur_axis = ax[c // fig_nrow, c % fig_nrow]
            pxb_obj = cur_axis.bxp(
                    bxpstats= plot_data,
                    showfliers=False,
                    shownotches=False,
                    showmeans=False,
                    boxprops = boxprops,
                    patch_artist=True,
                    positions = positions,
                )
            cur_axis.set_title(self.deconvolution.dataset.cell_type_str_list[c])
            cur_axis.set_xticklabels(
                list(self.deconvolution.dataset.bulk_sample_names[x.item()] for x in sort_order), 
                rotation = 90)

        fig.show()
        fig.tight_layout()    
