import sys
sys.path.append('..')

import arviz as az
import argparse
import emcee
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from corner import corner 

from src.simulator import BurstSimulator, Model

from src.MCMC.helpers import gen_parameter_labels
from src.MCMC.plotting import plot_1d_hist, plot_posterior_samples
from src.MCMC.posterior import log_posterior
from src.MCMC.priors import UniformPrior

# prevent numpy slow down when parallelizing MCMC
import os
os.environ["OMP_NUM_THREADS"] = "1"

def main(nwalkers, burn_steps, steps, parallel):
    """
    Run MCMC sampler.
    """
    time = np.linspace(0, 1.0, 1000)
    
    # TODO: get via args
    N = 3
    inf_params = ["t0", "skew"]

    amp  = [20.0 for _ in range(N)]
    t0   = np.sort(np.random.rand(N))
    rise = [0.03 for _ in range(N)]
    skew = [5.0 for _ in range(N)]

    burstparams = {
        't0'   : t0,
        'amp'  : amp,
        'rise' : rise,
        'skew' : skew
        }
    print(f"Doing inference on a burst with {N} components and burstparams: {burstparams}")
    ybkg = 5.0

    model     = Model(time, N, burstparams, ybkg)
    simulator = BurstSimulator(model)
    simulated_counts = simulator.simulate_burst()

    true_values = np.array([simulator.get_true(key) for key in inf_params])
    true_values = true_values.flatten()

    modelparams = {
        'time' : time,
        'ncomp': N,
        'burstparams': burstparams,
        'ybkg': ybkg
    }

    # define the priors
    prior_dict = {
        "t0"  : UniformPrior(x_min=0,    x_max=1,   log=False, enforce_order=True, dim=N),
        "amp" : UniformPrior(x_min=10,   x_max=300, log=False,  enforce_order=False, dim=N),
        "rise": UniformPrior(x_min=1e-4, x_max=0.1,  log=True, enforce_order=False, dim=N),
        "skew": UniformPrior(x_min=1,    x_max=6,   log=False, enforce_order=False, dim=N)
    }

    # https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    pool = Pool() if parallel else None
    ndim = len(inf_params) * N
    p0 = [prior_dict[key].sample(nwalkers) for key in inf_params]
    p0 = np.concatenate(p0, axis=1)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[inf_params, prior_dict, modelparams, simulated_counts], pool=pool)

    # burn-in
    state = sampler.run_mcmc(p0, burn_steps)
    sampler.reset()

    # running the sampler
    sampler.run_mcmc(state, steps, progress=True)

    if parallel:
        pool.close()
        pool.join()

    # evaluating the results
    samples = sampler.get_chain(flat=True)

    # plot the OG
    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    simulator.plot_burst()

    # samples from posterior overlayed on true model
    plt.subplot(122)
    N_samples = 100
    plot_posterior_samples(N_samples, simulated_counts, samples, inf_params, modelparams)
    plt.tight_layout()

    # corner plot (or histogram for 1D)
    var_names = gen_parameter_labels(inf_params, N)
    if ndim > 1:
        fig = corner(samples, labels=var_names, truths=true_values)
    else:
        plt.figure()
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