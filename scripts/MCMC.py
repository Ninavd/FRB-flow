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

def log_likelihood(t0: float, modelparams, simulated_counts: Iterable[int]) -> float:
    """
    $ln(p(x_{counts} | t0)$
    natural logarithm of poisson likelihood.
    """
    
    L = 0
 
    # update model parameters
    modelparams['burstparams']['t0'] = [t0]

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

def log_prob(t0: Iterable[float], *args) -> float:
    """
    Log of the posterior distribution.
    """
    # t0 is bounded by the uniform prior
    if t0 < 0 or t0 > 1:
        return -np.inf
    
    # otherwise its equal to the likelihood
    return log_likelihood(t0, *args)

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

    for i in range(100):
        random_index = np.random.randint(low=0, high=len(samples))
        random_sample = samples[random_index]
        modelparams['burstparams']['t0'] = random_sample
        model = Model(**modelparams).get_flux()
        plt.plot(time, model, alpha=0.5, label=f"{'posterior samples' if i == 0 else ''}", color='gray')

    plt.title('100 posterior samples')
    plt.legend()

def main(nwalkers, burn_steps, steps, parallel):
    """
    Run MCMC sampler.
    """
    time = np.linspace(0, 1.0, 1000)
    N = 1

    amp  = 100.0
    t0   = 0.4
    rise = 0.03
    skew = 5.0

    burstparams = {
        't0'   : [t0],
        'amp'  : [amp],
        'rise' : [rise],
        'skew' : [skew]
    }

    ybkg = 5.0

    model     = Model(time, N, burstparams, ybkg)
    simulator = BurstSimulator(model)
    noise_free_model, simulated_counts = simulator.simulate_burst(return_model=True)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    simulator.plot_burst()

    true_t0 = simulator.get_true('t0')

    modelparams = {
        'time' : time,
        'ncomp': N,
        'burstparams': burstparams,
        'ybkg': ybkg
    }

    # https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    pool = Pool() if parallel else None
    ndim     = 1 # TODO: infer from some kind of argument
    p0       = np.random.rand(nwalkers, ndim) # initial position of the walkers
    sampler  = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[modelparams, simulated_counts], pool=pool)

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

    # corner plot (histogram for 1D)
    plt.subplot(133)
    if ndim > 1:
        fig = corner(samples, labels=["t0_1", "t0_2"], truths=true_t0)
    else:
        plot_1d_hist(samples, true_t0)

    plt.tight_layout()
    plt.show()

    # trace plot (with arviz?)
    inference_data = az.from_emcee(sampler, var_names=["t0"]) # TODO: Make names general
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