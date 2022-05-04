"""Helper functions for command-line functionality"""

import torch
import numpy as np
import anndata
from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel
from ternadecov.dataset import DeconvolutionDataset
from ternadecov.parametrization import (
    DeconvolutionDatatypeParametrization,
    DeconvolutionDatasetParametrization,
    TimeRegularizedDeconvolutionModelParametrization,
    TimeRegularizedDeconvolutionGPParametrization,
)
from ternadecov.deconvolution_exporter import DeconvolutionExporter


def get_torch_device(args):
    """Get the torch device based on availability and cli params"""
    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            if args.verbose:
                print("Using CUDA")
        else:
            # quietly use cpu
            device = torch.device("cpu:0")
    else:
        device = torch.device("cpu:0")

    return device


def do_deconvolution(args):
    """Main function that processes and executes deconvolution sub-command
    
    :param args: parsed cli params
    """

    device = get_torch_device(args)

    dtype = torch.float32
    dtype_np = np.float32

    bulk_anndata_path = args.bulk_anndata
    sc_anndata_path = args.sc_anndata

    if args.verbose:
        print("Loading bulk data...")
    with open(bulk_anndata_path, "rb") as fh:
        bulk_anndata = anndata.read_h5ad(fh)

    if args.verbose:
        print("Loading single-cell data...")
    with open(sc_anndata_path, "rb") as fh:
        sc_anndata = anndata.read_h5ad(fh)

    if args.verbose:
        print("Preparing deconvolution...")

    # Use the default datatype parametrization
    datatype_param = DeconvolutionDatatypeParametrization()

    # Prepare the dataset
    dataset = DeconvolutionDataset(
        types=datatype_param,
        parametrization=DeconvolutionDatasetParametrization(
            sc_anndata=sc_anndata,
            sc_celltype_col=args.sc_celltype_column,
            bulk_anndata=bulk_anndata,
            bulk_time_col=args.bulk_time_column,
            feature_selection_method=args.feature_selection_method,
        ),
    )
    if args.verbose:
        print("Running deconvolution...")
    deconvolution = TimeRegularizedDeconvolutionModel(
        dataset=dataset,
        trajectory_model_type="gp",
        hyperparameters=TimeRegularizedDeconvolutionModelParametrization(),
        trajectory_hyperparameters=TimeRegularizedDeconvolutionGPParametrization(),
        types=datatype_param,
    )
    deconvolution.fit_model(
        n_iters=args.iterations, verbose=args.verbose, log_frequency=args.log_frequency
    )

    # Export the results
    if args.verbose:
        print("Saving results...")
    exporter = DeconvolutionExporter(deconvolution, prefix=args.export_prefix)
    exporter.export_results(args.export_directory)
