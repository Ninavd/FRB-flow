import sys
sys.path.append('..')

import arviz as az
import argparse
import emcee
import numpy as np
import math 
import matplotlib.pyplot as plt

from multiprocessing import Pool
from typing import Iterable
from corner import corner 

from src.simulator import BurstSimulator, Model

# prevent numpy slow down when parallelizing MCMC
import os
os.environ["OMP_NUM_THREADS"] = "1"

def log_likelihood(theta: Iterable[float], inf_params, modelparams, simulated_counts: Iterable[int]) -> float:
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
    N = modelparams['ncomp']
    for i, key in enumerate(inf_params):
        start = i * N
        stop  = start + N

        modelparams['burstparams'][key] = theta[start:stop]

    # find noise-free flux
    model_counts = Model(**modelparams).get_flux()
    
    # likelihood of the observed counts under this model
    L = (
        -1 * np.sum(np.log(model_counts)) 
        + np.sum(simulated_counts * np.log(model_counts)) 
        - np.sum([math.log(math.factorial(x)) for x in simulated_counts]) # Note x is simulated count and should always be integer
    )
    
    # TODO: Compare w/ likelihood function of Daniela
    return L

class UniformPrior():
    """
    Class for uniform prior.
    """
    def __init__(self, x_min, x_max, log, enforce_order):
        self.x_min = x_min
        self.x_max = x_max
        self.log = log
        self.enforce_order = enforce_order

    def log_prob(self, x):
        # out of range
        if x.any() < self.x_min or x.any() > self.x_max:
            return -np.inf
        
        # check if first value should be smaller than second etc. etc.
        if self.enforce_order and not np.all(x[:-1] <= x[1:]):
            return -np.inf
        
        # log uniform 
        if self.log:
            return -np.log(x) - np.log(np.log(self.x_max) - np.log(self.x_min))
        
        else:
            return 1

def log_priors(theta, *args):
    """
    Find the value of the prior encompassing all the parameters.
    """
    inf_params, modelparams, _ = args 

    # TODO: make global var or add to args
    prior_dict = {
        "t0"  : UniformPrior(x_min=0,    x_max=1,   log=False, enforce_order=True),
        "amp" : UniformPrior(x_min=10,   x_max=300, log=True,  enforce_order=False),
        "rise": UniformPrior(x_min=1e-4, x_max=0.1,  log=True, enforce_order=False),
        "skew": UniformPrior(x_min=1,    x_max=6,   log=False, enforce_order=False)
    }
    
    N = modelparams['ncomp']
    log_prob = 0
    for i, key in enumerate(inf_params):
        start = i * N
        stop  = start + N

        param = theta[start:stop]
        prior = prior_dict[key]
        log_prob += prior.log_prob(param)

        if log_prob == -np.inf:
            break
    
    return log_prob

def log_prob(theta: Iterable[float], *args) -> float:
    """
    Log of the posterior distribution.
    """
    log_priors(theta, *args)
    
    # otherwise its equal to the likelihood
    return log_likelihood(theta, *args)


def plot_1d_hist(samples, true_value):
    """
    Plots histogram of (1D) sample distribution.
    """
    frequencies, _, _ = plt.hist(samples[:, 0], 50, color="k", histtype="step")
    plt.vlines(true_value,  ymin=0, ymax=max(frequencies), linestyles="dashed", color ='red', label='true $t_0$')

    plt.title("Sample distribution")
    plt.xlabel("$t_0$")
    plt.ylabel("$p(t_0 | x_0)$")
    plt.gca().set_yticks([])

    plt.legend()

def plot_posterior_samples(N, simulated_counts, samples, modelparams):
    """
    Overlay samples from posterior on simulated curve.
    """
    time = modelparams["time"]
    plt.plot(time, simulated_counts, 'k-', alpha=0.3, label="data")

    for i in range(N):
        random_index = np.random.randint(low=0, high=len(samples))
        random_sample = samples[random_index]
        modelparams['burstparams']['t0'] = random_sample
        model = Model(**modelparams).get_flux()
        plt.plot(time, model, alpha=0.5, label=f"{'posterior samples' if i == 0 else ''}", color='gray')

    plt.title(f'{N} posterior samples')
    plt.legend()

def gen_parameter_labels(inf_params, N) -> Iterable[str]:
    """
    Generates parameter labels of shape t0_1, t0_2, amp_1, amp_2, etc.
    """
    labels = []
    for key in inf_params:
        for i in range(N):
            label = f"{key}_{i+1}"
            labels.append(label)

    return labels

def main(nwalkers, burn_steps, steps, parallel):
    """
    Run MCMC sampler.
    """
    time = np.linspace(0, 1.0, 1000)
    
    # TODO: get via args
    N = 2
    inf_params = ["t0"]

    amp  = [100.0 for _ in range(N)]
    t0   = np.sort(np.random.rand(N))
    rise = [0.03 for _ in range(N)]
    skew = [5.0 for _ in range(N)]

    burstparams = {
        't0'   : t0,
        'amp'  : amp,
        'rise' : rise,
        'skew' : skew
        }

    ybkg = 5.0

    model     = Model(time, N, burstparams, ybkg)
    simulator = BurstSimulator(model)
    noise_free_model, simulated_counts = simulator.simulate_burst(return_model=True)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    simulator.plot_burst()

    true_values = [simulator.get_true(key) for key in inf_params]

    modelparams = {
        'time' : time,
        'ncomp': N,
        'burstparams': burstparams,
        'ybkg': ybkg
    }

    # https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    pool = Pool() if parallel else None
    ndim = len(inf_params) * N
    p0       = np.random.rand(nwalkers, ndim) # initial position of the walkers TODO: p0 must obey the respective priors
    
    # enforce order if peaktime is included in inference params
    if "t0" in inf_params:
        idx = inf_params.index("t0")
        p0[:, idx * N: idx * N + N] = np.sort(np.random.rand(nwalkers, ndim), axis=1)

    sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[inf_params, modelparams, simulated_counts], pool=pool)

    # burn-in
    state      = sampler.run_mcmc(p0, burn_steps)
    sampler.reset()

    # running the sampler
    sampler.run_mcmc(state, steps, progress=True)

    if parallel:
        pool.close()
        pool.join()

    # evaluating the results
    samples = sampler.get_chain(flat=True)

    # samples from posterior overlayed on true model
    plt.subplot(132)
    N_samples = 100
    plot_posterior_samples(N_samples, simulated_counts, samples, modelparams)

    # corner plot (or histogram for 1D)
    var_names = gen_parameter_labels(inf_params, N)
    if ndim > 1:
        plt.figure()
        fig = corner(samples, labels=var_names, truths=true_values)
    else:
        plt.subplot(133)
        plot_1d_hist(samples, true_values)

    plt.tight_layout()
    plt.show()

    # trace plot with arviz
    inference_data = az.from_emcee(sampler, var_names) 
    az.plot_trace(inference_data, compact=True)
    plt.tight_layout()
    plt.show()

    # print some stats
    print(az.summary(inference_data))
    print("Autocorrelation time: ", sampler.get_autocorr_time())
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling algorithm")

    parser.add_argument("-w", "--walkers", type=int, default=20,   help="Number of walkers")
    parser.add_argument("-s", "--steps",   type=int, default=1000, help="Number of samples (steps per walker)")
    parser.add_argument("-b", "--burn",    type=int, default=300,  help="burn steps, number of burn-in steps used")
    parser.add_argument("--fast", action="store_true", help="speed up sampling by using parallelization")

    args = parser.parse_args()
    main(args.walkers, args.burn, args.steps, args.fast)