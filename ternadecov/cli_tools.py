import torch

from ternadecov.time_deconv import *
from ternadecov.stats_helpers import *
from ternadecov.time_deconv_simulator import *
from ternadecov.stats_helpers import *
from ternadecov.hypercluster import *
from ternadecov.dataset import *
from ternadecov.trajectories import *

def get_torch_device(args):
    # Check for cuda device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")

def do_deconvolution(args):
    device = get_torch_device(args)
        
    dtype = torch.float32
    dtype_np = np.float32
    
    bulk_anndata_path = args.bulk_anndata
    sc_anndata_path = args.sc_anndata
    
    if args.verbose:
        print('Loading bulk data...')
    with open(bulk_anndata_path, 'rb') as fh:
        bulk_anndata  = anndata.read_h5ad(fh)
        
    if args.verbose:
        print('Loading single-cell data...')
    with open(sc_anndata_path, 'rb') as fh:
        sc_anndata = anndata.read_h5ad(fh)
        
    if args.verbose:
        print('Preparing deconvolution...')
    ebov_dataset = DeconvolutionDataset(
        sc_anndata = sc_anndata,
        sc_celltype_col = args.sc_celltype_column,
        bulk_anndata = bulk_anndata,
        bulk_time_col = args.bulk_time_column,
        dtype_np = dtype_np,
        dtype = dtype,
        device=device,
        feature_selection_method = args.feature_selection_method,
        hypercluster = args.hypercluster,
    )
    
    pseudo_time_reg_deconv = TimeRegularizedDeconvolution(
        dataset=ebov_dataset,
        polynomial_degree = args.polynomial_degree,
        basis_functions = args.basis_function_type,
        device=device,
        dtype=dtype
    )
    
    pseudo_time_reg_deconv.fit_model(n_iters=args.iterations, verbose=args.verbose, log_frequency=args.log_frequency)
    
    pseudo_time_reg_deconv.write_sample_compositions(args.sample_output_csv)