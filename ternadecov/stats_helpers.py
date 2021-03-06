"""Statistics helper functions"""

import functools
import pyro.distributions as dist
import torch
from scipy.special import legendre


@functools.lru_cache(maxsize=20)
def legendre_coefficient_mat(k_max, dtype, epsilon=1e-8):
    """
    Return the coefficient matrix of legendre polynomials.

    :param k_max: legenre polynomial max degree
    :param epsilon: minimum coefficient value
    
    :return: torch tensor with legendre coefficients
    """

    k_max_internal = k_max + 1
    X_kl = torch.zeros(k_max_internal, k_max_internal)
    for i in range(0, k_max_internal):
        terms = list(legendre(i))
        terms.reverse()
        X_kl[i, : i + 1] = torch.tensor(terms, dtype=dtype)
    X_kl = torch.where(X_kl.abs() < epsilon, torch.zeros_like(X_kl), X_kl)
    return X_kl


def NegativeBinomialAltParam(mu, phi):
    """Creates a negative binomial distribution.

    :param mu: mean (must be strictly positive)
    :param phi: overdispersion (must be strictly positive)
        
    :return: pyro distribution
    """
    return dist.GammaPoisson(concentration=1 / phi, rate=1 / (mu * phi))
