class TimeRegularizedDeconvolutionModelParametrization:
    def __init__(self):
        # Prior
        self.log_beta_prior_scale = 1.0
        self.tau_prior_scale = 1.0
        self.log_phi_prior_loc = -5.0
        self.log_phi_prior_scale = 1.0
        
        # Posterior
        self.init_posterior_global_scale_factor = 0.05
        
        self.log_beta_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.tau_posterior_scale = 1.0 * self.init_posterior_global_scale_factor
        self.log_phi_posterior_loc = -5.0
        self.log_phi_posterior_scale = 0.1 * self.init_posterior_global_scale_factor
        
class TimeRegularizedDeconvolutionGPParametrization:
    def __init__(self):
        
        self._num_inducing_points = 10  # 10 50 100
        
        self.init_rbf_kernel_lengthscale = 0.5 # .1, 1
        self.init_rbf_kernel_variance = 0.5 # .1, 1
        self.init_whitenoise_kernel_variance = 0.1 # .1, , 1/2,  1
        self.gp_cholesky_jitter = 1e-4
    
    @property
    def num_inducing_points(self):
        return self._num_inducing_points
    
    @num_inducing_points.setter
    def num_inducing_points(self, value):
        self._num_inducing_points = int(value)