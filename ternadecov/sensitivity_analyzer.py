from ternadecov.time_deconv import *
from ternadecov.simulator import *
from ternadecov.stats_helpers import *
from ternadecov.deconvolution_plotter import *
from ternadecov.parametrization import *

import logging


class SensitivityAnalyzer:
    """Container class for static methods pertaining to parameter sensitivity analysis"""

    @staticmethod
    def evaluate_deconvolution(
        dataset_param: DeconvolutionDatasetParametrization,
        hyperparameters: TimeRegularizedDeconvolutionModelParametrization,
        trajectory_hyperparameters: TimeRegularizedDeconvolutionGPParametrization,
        datatype_param: DeconvolutionDatatypeParametrization,
        n_iters=1_000,
    ):
        """Evaluate deconvolution with specified parametrization"""

        dataset = DeconvolutionDataset(
            parametrization=dataset_param, types=datatype_param,
        )

        # Make the deconvolution model
        model = TimeRegularizedDeconvolutionModel(
            dataset=dataset,
            trajectory_model_type="gp",
            hyperparameters=hyperparameters,
            trajectory_hyperparameters=trajectory_hyperparameters,
            types=datatype_param,
        )

        model.fit_model(
            n_iters=n_iters,
            verbose=False,
            log_frequency=1000,
            keep_param_store_history=False,
            clear_param_store=True,
        )

        traj = model.population_proportion_model.get_composition_trajectories(
            model.dataset, n_intervals=100
        )

        return traj

    @staticmethod
    def scan_parameter(
        parameter,
        sc_anndata,
        bulk_anndata,
        datatype_param,
        parameter_type="model",
        parameter_variable_type="continuous",
        start=None,
        end=None,
        num=None,
        discrete_values=None,
        n_iters=10_000,
    ):
        """Scan the defined parameter with values in the specified range and save results"""
        results = {}

        if parameter_variable_type == "continuous":
            assert start is not None
            assert end is not None
            assert num is not None
            param_values = np.linspace(start, end, num)
        elif parameter_variable_type == "discrete":
            assert discrete_values is not None
            param_values = discrete_values
        else:
            raise ValueError(
                f"Unknown parameter_variable_type: {parameter_variable_type}"
            )

        # evaluation loop
        for v in param_values:
            logging.info(f"Evaluating with {parameter} = {v} ...")

            # Preparing parameters
            dataset_param = DeconvolutionDatasetParametrization(
                sc_anndata=sc_anndata,
                sc_celltype_col="Subclustering_reduced",
                bulk_anndata=bulk_anndata,
                bulk_time_col="dpi_time",
            )
            model_param = TimeRegularizedDeconvolutionModelParametrization()
            trajectory_param = TimeRegularizedDeconvolutionGPParametrization()

            # Modifying parameters
            if parameter_type == "model":
                setattr(model_param, parameter, v)
            elif parameter_type == "trajectory":
                setattr(trajectory_param, parameter, v)
            elif parameter_type == "dataset":
                setattr(dataset_param, parameter, v)
            else:
                raise ValueError(f"Unknown parameter_type: {parameter_type}")

            # Evaluating
            results[v] = SensitivityAnalyzer.evaluate_deconvolution(
                dataset_param=dataset_param,
                hyperparameters=model_param,
                trajectory_hyperparameters=trajectory_param,
                n_iters=n_iters,
                datatype_param=datatype_param,
            )

        return results

    @staticmethod
    def plot_scan_trajectories(results, variable):
        """Plot the results of scan_parameter"""

        r = results[variable]

        keys = list(r.keys())
        n = len(keys)
        nr = math.ceil(math.sqrt(n))
        nc = math.floor(math.sqrt(n))

        fig, ax = matplotlib.pyplot.subplots(nr, nc, figsize=(10, 10))

        for k in range(len(keys)):
            traj = r[keys[k]]
            cur_ax = ax[k % nr, k // nr]
            cur_ax.plot(
                traj["true_times_z"], traj["norm_comp_tc"],
            )
            cur_ax.set_title(f"{variable} = {keys[k]}")
            cur_ax.set_xlabel("Time")

        fig.tight_layout()
