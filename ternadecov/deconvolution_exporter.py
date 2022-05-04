"""Deconvolution exporter object for automated export"""


from ternadecov.deconvolution_plotter import DeconvolutionPlotter
from ternadecov.deconvolution_writer import DeconvolutionWriter
from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel


class DeconvolutionExporter:
    """Class for automated exporting of the deconvolution results"""

    def __init__(self, deconvolution: TimeRegularizedDeconvolutionModel, prefix=""):
        """Initializer for DeconvolutionExporter
        
        :param self: An instance of class
        :param deconvolution: A deconvolution object to plot data for 
        :param prefix: filename prefix to use in all exporting
        """

        self.deconvolution = deconvolution
        self.plotter = DeconvolutionPlotter(self.deconvolution)
        self.writer = DeconvolutionWriter(self.deconvolution)
        self.prefix = prefix

    def export_results(
        self, output_directory, save_pdf=True, save_png=True, save_csv=True
    ):
        """Export all the results
        
        :param self: An instance of object
        :param output_directory: A directory location to export the results to 
        :param save_pdf: Save the figures as PDF?
        :param save_png: Save the figures as PNG?
        :param save_csv: Save the numerical output as CSV
        """

        extensions = []
        if save_pdf:
            extensions.append(".pdf")
        if save_png:
            extensions.append(".png")

        def construct_filenames(plot_name, extensions):
            """Helper function for constructing filesnames for export"""

            loss_plot_filename_prefix = f"{output_directory}/{self.prefix}{plot_name}"
            loss_filenames = (loss_plot_filename_prefix + x for x in extensions)
            return loss_filenames

        self.plotter.plot_loss(filenames=construct_filenames("_loss", extensions))

        self.plotter.plot_phi_g_distribution(
            filenames=construct_filenames("_phi_g", extensions)
        )

        self.plotter.plot_beta_g_distribution(
            filenames=construct_filenames("_beta_g", extensions)
        )

        self.plotter.plot_sample_compositions_scatter(
            filenames=construct_filenames("_scatter", extensions)
        )

        self.plotter.plot_sample_compositions_boxplot(
            filenames=construct_filenames("_boxplot", extensions)
        )

        self.plotter.plot_sample_compositions_boxplot_confidence(
            verbose=False,
            spacing=3,
            filenames=construct_filenames("_boxplot_confidence", extensions),
        )

        # One panel per celltype with error
        self.plotter.plot_composition_trajectories_via_posterior_sampling(
            iqr_alpha=0.2,
            show_combined=False,
            ncols=3,
            lw=2.0,
            figsize=(6, 6),
            sharey=False,
            filenames=construct_filenames(
                "_trajectories_posterior_sampling", extensions
            ),
        )

        # All celltypes in one panel, no errorbars
        self.plotter.plot_composition_trajectories_via_posterior_sampling(
            iqr_alpha=0.0,
            show_combined=True,
            ncols=3,
            lw=2.0,
            figsize=(8, 8),
            sharey=False,
            filenames=construct_filenames(
                "_trajectories_posterior_sampling_combined", extensions
            ),
        )

        if save_csv:
            self.writer.write_cell_compositions(
                filename=f"{output_directory}/{self.prefix}_compositions.csv"
            )
            self.writer.write_sample_draws_quantiles(
                filename=f"{output_directory}/{self.prefix}_sample_quantile_compositions.csv"
            )
