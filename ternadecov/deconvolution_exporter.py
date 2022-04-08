from ternadecov.deconvolution_plotter import DeconvolutionPlotter
from ternadecov.deconvolution_writer import DeconvolutionWriter
from ternadecov.time_deconv import TimeRegularizedDeconvolutionModel

class DeconvolutionExporter():
    def __init__(self, deconvolution: TimeRegularizedDeconvolutionModel, prefix=""):
        self.deconvolution = deconvolution
        self.plotter = DeconvolutionPlotter(self.deconvolution)
        self.writer = DeconvolutionWriter(self.deconvolution)
        self.prefix = prefix
        
    def export_results(self, output_directory, save_pdf = True, save_png = True, save_csv = True):
        
        extensions = []
        if save_pdf:
            extensions.append('.pdf')
        if save_png:
            extensions.append('.png')

        def construct_filenames(plot_name, extensions):     
            loss_plot_filename_prefix = f'{output_directory}/{self.prefix}{plot_name}'
            loss_filenames = ( loss_plot_filename_prefix + x for x in extensions )
            return loss_filenames

        self.plotter.plot_loss(
            filenames=construct_filenames('_loss', extensions)
        )
        
        self.plotter.plot_composition_trajectories(
            filenames=construct_filenames('_trajectories', extensions)
        )
        
        self.plotter.plot_gp_composition_trajectories(
            filenames=construct_filenames('_gp_composition', extensions)
        )
        
        self.plotter.plot_phi_g_distribution(
            filenames=construct_filenames('_phi_g', extensions)
        )
        
        self.plotter.plot_beta_g_distribution(
            filenames=construct_filenames('_beta_g', extensions)
        )
        
        self.plotter.plot_sample_compositions_scatter(
            filenames=construct_filenames('_scatter', extensions)
        )
        
        self.plotter.plot_sample_compositions_boxplot(
            filenames=construct_filenames('_boxplot', extensions)
        )
        
        self.plotter.plot_sample_compositions_boxplot_confidence(
            verbose=False,
            spacing=3,
            filenames=construct_filenames('_boxplot_confidence', extensions)
        )
        
        if save_csv:
                self.writer.write_cell_compositions(filename=f'{output_directory}/{self.prefix}_compositions.csv')
                self.writer.write_sample_draws_quantiles(filename=f'{output_directory}/{self.prefix}_sample_quantile_compositions.csv')