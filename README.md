# Temporal RNA-seq Deconvolution (TeRNAdecov)

TeRNAdecov is a package for deconvolution of bulk RNA-seq samples from time series using single-cell datasets. It allows for simultaneous inference of sample-specific composition as well as overall time trajectory dynamics using highly scalable stochastic variational inference.

## Install
Installation in an Anaconda environment is recommended.

Install pyro (https://pyro.ai/) and pytorch 

Clone the repository
```sh
git clone git@github.com:broadinstitute/temporal-rna-seq-deconvolution
cd temporal-rna-seq-deconvolution
```

In ubuntu you might need to install the following for the packages to compile
```sh
sudo apt-get install libglib2.0-dev python3-dev
```

Install python package
```py
pip install -e .
```

## Documentation
Auto-generated code documentation for the package can be found on [readthedocs.io](https://ternadecov.readthedocs.io/en/latest/source/ternadecov.html).

## Tutorials
For a full tutorial on the use of ternadecov please visit the [GP deconvolution tutorial](notebooks/tutorials/tutorial-deconvolve-gp.ipynb).

Additional tutorials provide information on:
* [Simulating and saving datasets](notebooks/tutorials/tutorial-simulate-save.ipynb)
* [Simulating linear trajectories](notebooks/tutorials/tutorial-deconvolve-simulated-linear.ipynb)
* [Simulating sigmoid trajectories](notebooks/tutorials/tutorial-deconvolve-simulated-sigmoid.ipynb)
* [Simulating periodic trajectories](notebooks/tutorials/tutorial-deconvolve-simulated-periodic.ipynb)
* [Evaluating the time required to run the deconvolution for different numbers of samples](notebooks/tutorials/tutorial-evaluate-run-time-gp.ipynb)

## CLI Use
ternadecov offers an cli for automated deconvolution and exporting of results from the command line. An example run call could be:

```bash

ternadecov deconvolve \
  --bulk-anndata bulk.h5ad \
  --sc-anndata singlecell.h5ad \
  --iterations 20000 \
  --sc-celltype-column Abbreviation \
  --bulk-time-column dpi_time \
  --feature-selection-method overdispersed_bulk_and_high_sc \
  --export-prefix results_ \
  --export-directory output/
```

For more information on CLI usage try:

```bash
ternadecov --help
```
