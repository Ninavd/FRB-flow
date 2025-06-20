import numpy as np

from typing import Iterable

from src.simulator import Model
from src.helpers import update_modelparams

def log_likelihood(theta: Iterable[float], inf_params, priors, modelparams, simulated_counts: Iterable[int]) -> float:
    """
    $ln(p(x_{counts} | t0)$
    natural logarithm of poisson likelihood.

    Args:
        theta:       vector with sampled inference parameters
        inf_params:  keys of inference parameters, f.e. [t0, amp]
        modelparams: dictionary of all model parameters used to generate original burst
        simulated_counts: binned counts of simulated counts
    """
    
    # extract parameters from theta to update model params
    modelparams = update_modelparams(theta, inf_params, modelparams)

    # find noise-free flux
    model_counts = Model(**modelparams).get_flux()
    
    # likelihood of the observed counts under this model
    L = (
        -1 * np.sum(model_counts) 
        + np.sum(simulated_counts * np.log(model_counts)) 
        # The term below is left out since it does not affect optimization (independent of modelparams)
        #- np.sum([math.log(math.factorial(x)) for x in simulated_counts]) # Note x is simulated count and should always be integer
    )
    
    # TODO: Compare w/ likelihood function of Daniela
    return L

def log_priors(theta, *args):
    """
    Find the value of the prior encompassing all the parameters.
    """
    inf_params, priors, modelparams, _ = args 
    
    N = modelparams['ncomp']
    log_prob = 0
    for i, key in enumerate(inf_params):
        start = i * N
        stop  = start + N

        param = theta[start:stop]
        prior = priors[key]
        log_prob += prior.log_prob(param)

        if log_prob == -np.inf:
            break
    
    return log_prob

def log_posterior(theta: Iterable[float], *args) -> float:
    """
    Log of the posterior distribution.
    """
    log_prior_value = log_priors(theta, *args)

    # check if samples are valid via prior
    if log_prior_value == -np.inf:
        return -np.inf
    
    # if so, it's safe to find the likelihood
    return log_prior_value + log_likelihood(theta, *args)