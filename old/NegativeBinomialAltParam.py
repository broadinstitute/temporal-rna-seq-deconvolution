from pyro.distributions.torch_distribution import (
    TorchDistribution,
    TorchDistributionMixin,
)
from torch.distributions.utils import (
    probs_to_logits,
    logits_to_probs,
    broadcast_all,
    lazy_property,
)
from torch.distributions import constraints
from numbers import Number
import torch


class NegativeBinomialAltParam(TorchDistribution):
    r"""
    Creates a negative binomial distribution.

    Args:
        mu (Number, Tensor): mean (must be strictly positive)
        phi (Number, Tensor): overdispersion (must be strictly positive)
    """
    arg_constraints = {"mu": constraints.positive, "phi": constraints.positive}
    support = constraints.nonnegative_integer
    EPS = 1e-6

    def __init__(self, mu, phi, validate_args=None):
        self.mu, self.phi = broadcast_all(mu, phi)
        if all(isinstance(_var, Number) for _var in (mu, phi)):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(NegativeBinomialAltParam, self).__init__(
            batch_shape, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeBinomialAltParam, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.phi = self.phi.expand(batch_shape)

        super(NegativeBinomialAltParam, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def _gamma(self):
        return torch.distributions.Gamma(
            concentration=self.phi.reciprocal(), rate=(self.mu * self.phi).reciprocal()
        )

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return torch.poisson(self._gamma.sample(sample_shape=sample_shape))

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mu + self.phi * self.mu.pow(2)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        mu, phi, value = broadcast_all(self.mu, self.phi, value)
        alpha = (self.EPS + phi).reciprocal()
        return (
            (value + alpha).lgamma()
            - (value + 1).lgamma()
            - alpha.lgamma()
            + alpha * (alpha.log() - (alpha + mu).log())
            + value * (mu.log() - (alpha + mu).log())
        )
