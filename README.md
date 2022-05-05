# Temporal RNA-seq Deconvolution (TeRNAdecov)

TeRNAdecov is a package for deconvolution of bulk RNA-seq samples from time series using single-cell datasets. It allows for simultaneous inference of sample-specific composition as well as overall time trajectory dynamics using highly scalable stochastic variational inference.

## Install
Clone the repository
```sh
git clone git@github.com:broadinstitute/temporal-rna-seq-deconvolution
cd temporal-rna-seq-deconvolution
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
