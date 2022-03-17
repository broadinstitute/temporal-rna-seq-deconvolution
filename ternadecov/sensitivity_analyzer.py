from ternadecov.time_deconv import *
from ternadecov.simulator import *
from ternadecov.stats_helpers import *
from ternadecov.deconvolution_plotter import *
from ternadecov.parametrization import *


class SensitivityAnalyzer:
    """Container class for static methods pertaining to parameter sensitivity analysis"""

    @staticmethod
    def evaluate_deconvolution(
        sc_anndata,
        bulk_anndata,
        hyperparameters: TimeRegularizedDeconvolutionModelParametrization,
        trajectory_hyperparameters: TimeRegularizedDeconvolutionGPParametrization,
        datatype_param: DeconvolutionDatatypeParametrization,
        n_iters=1_000,
    ):
        """Evaluate deconvolution with specified parametrization"""

        # Make the dataset
        dataset_param = DeconvolutionDatasetParametrization(
            sc_anndata=sc_anndata,
            sc_celltype_col="Subclustering_reduced",
            bulk_anndata=bulk_anndata,
            bulk_time_col="dpi_time",
        )

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
        start,
        end,
        num,
        sc_anndata,
        bulk_anndata,
        datatype_param,
        parameter_type="model",
        n_iters=10_000,
    ):
        """Scan the defined parameter with values in the specified range and save results"""
        results = {}

        for i, v in enumerate(np.linspace(start, end, num)):
            print(f"Evaluating with {parameter} = {v} ...")

            model_param = TimeRegularizedDeconvolutionModelParametrization()
            trajectory_param = TimeRegularizedDeconvolutionGPParametrization()

            if parameter_type == "model":
                setattr(model_param, parameter, v)
            elif parameter_type == "trajectory":
                setattr(trajectory_param, parameter, v)
            else:
                raise ValueError()

            results[v] = SensitivityAnalyzer.evaluate_deconvolution(
                sc_anndata=sc_anndata,
                bulk_anndata=bulk_anndata,
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
