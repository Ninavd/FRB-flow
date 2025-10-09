import sys
sys.path.append('..')

import arviz as az
import argparse
import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml 
import time as t

from multiprocessing import Pool
from corner import corner 
from copy import deepcopy
from datetime import datetime 

from src.simulator import BurstSimulator, Model
from src.helpers import plot_posterior_samples

from src.helpers import gen_parameter_labels
from src.MCMC.plotting import plot_1d_hist
from src.MCMC.posterior import log_posterior
from src.MCMC.priors import UniformPrior

# prevent numpy slow down when parallelizing MCMC
import os
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"

def main(N, inf_params, noise, nwalkers, burn_steps, max_steps,
         parallel, save, show_plots, FRB):
    """
    Run MCMC sampler.
    """
    # define the priors
    PRIORS = {
        "t0"  : UniformPrior(x_min=0.2,    x_max=0.8,   log=False, enforce_order=True, dim=N),
        "amp" : UniformPrior(x_min=1,   x_max=300, log=True,  enforce_order=False, dim=N),
        "rise": UniformPrior(x_min=1e-3, x_max=0.1,  log=True, enforce_order=False, dim=N),
        "skew": UniformPrior(x_min=1,    x_max=6,   log=False, enforce_order=False, dim=N)
    }

    time = np.linspace(0, 1.0, 1000)

    amp  = [float(np.log10(100)) for _ in range(N)]
    # amp  = [float(np.log10(100)), float(np.log10(34)), float(np.log10(50)), float(np.log10(89)), float(np.log10(50))]
    t0   = [float(t) for t in list(np.linspace(0.3, 0.7, N))]
    rise = [float(np.log10(0.03)) for _ in range(N)]
    skew = [2.0 for _ in range(N)]

    burstparams = {
        't0'   : t0,
        'amp'  : amp,
        'rise' : rise,
        'skew' : skew
        }
    # burstparams = {key:PRIORS[key].sample(1)[0] for key in PRIORS}
    # burstparams['amp'] = amp
    print(f"Doing inference on a burst with {N} component(s) and burstparams: {burstparams}")
    ybkg = 0.0

    model     = Model(time, N, deepcopy(burstparams), ybkg)
    simulator = BurstSimulator(model)
    simulated_counts = simulator.simulate_burst(noise=noise)

    true_values = np.array([burstparams[key] for key in inf_params])
    true_values = true_values.flatten()

    if FRB:
        df = pd.read_csv('../observational_data/FRBs/'+ FRB + '.dat', header=None, delimiter=' ')
        simulated_counts = np.array(df[0])
        simulator.simulated_counts = simulated_counts

    modelparams = {
        'time' : time,
        'ncomp': N,
        'burstparams': deepcopy(burstparams),
        'ybkg': ybkg
    }

    # https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    pool = Pool() if parallel else None
    ndim = len(inf_params) * N
    p0 = [PRIORS[key].sample(nwalkers) for key in inf_params]
    p0 = np.concatenate(p0, axis=1)
        
    sampler = emcee.EnsembleSampler(
        nwalkers, 
        ndim, 
        log_posterior,
        kwargs={'noise':noise}, 
        args=[inf_params, PRIORS, modelparams, simulated_counts], 
        pool=pool
        )

    # short run to determine best initial position
    print("_________________________________________")           
    print("\n1. RUN TO DETERMINE BEST INITIAL POSITION")
    print("_________________________________________\n") 
    start_time = t.time()
    state = sampler.run_mcmc(p0, burn_steps, progress=True)
    sampler.reset()

    # correct walkers with low likelihood (otherwise likely to get stuck)
    rounded_probs = 10 * np.floor(state.log_prob / 10)
    low_prob_mask = state.log_prob < np.max(rounded_probs)

    variations      = np.ones((sum(low_prob_mask), ndim)) * np.random.uniform(low=0.9, high=1.1, size=(sum(low_prob_mask), ndim))
    most_likely_pos = state.coords[np.argmax(state.log_prob)]
    state.coords[low_prob_mask] = most_likely_pos * variations

    print(f"\n Repositioned {sum(low_prob_mask)} of {nwalkers} walkers...")
    # print(state.coords)
    if sum(low_prob_mask) / nwalkers > 0.95:
        print("ERROR: Had to reposition more than 95% of walkers, increase number of burn steps or re-run")
        return 1

    # burn-in with repositioned walkers
    print("_____________________________")           
    print("\n          2. BURN-IN")
    print("_____________________________\n") 
    state = sampler.run_mcmc(state.coords, 500, progress=True)
    sampler.reset()

    # run until convergence
    print("_____________________________")           
    print("\n         3. FINAL RUN")
    print("_____________________________\n") 

    old_tau = np.inf
    N_checks = 0
    for state in sampler.sample(state.coords, iterations=max_steps, progress=True):
        
        # only check every 100 steps
        if sampler.iteration % 100 != 0:
            continue

        # using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)

        # Check convergence (from https://emcee.readthedocs.io/en/stable/tutorials/monitor/)
        # we assume convergence if the chain is longer than 100 x the estimated autocorrelation time 
        # and if this estimate changed by less than 1%
        converged = np.all(tau * 80 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
        N_checks += 1
        if converged:
            break

        old_tau = tau
        # print(old_tau)
        # convergence failed
        if sampler.iteration == max_steps and not converged:
            print(f"\n ConvergenceError: Convergence condition not satisfied within {max_steps} steps. Try increasing the max_steps argument or re-run.")
            return
    run_time = t.time() - start_time
    print(f"\n -> Converged after {sampler.iteration} steps \n")
  
    if parallel:
        pool.close()
        pool.join()

    # set up save
    if save:
        
        # make run dir
        timestamp = datetime.today().strftime('%d-%m_%H:%M:%S')
        save_dir = "../MCMC_runs/run_" + timestamp
        os.makedirs(save_dir)
        
        for key in burstparams:
            burstparams[key] = list(burstparams[key])

        # save settings yaml
        settings = {
            "N":N,
            "inf_params":inf_params,
            "burstparams":burstparams,
            "runtime": run_time,
            "iterations": sampler.iteration
        }
        with open(os.path.join(save_dir, "settings.yaml"), "w") as f:
            yaml.dump(settings, f, sort_keys=False)

        # save simulated lightcurve
        np.save(os.path.join(save_dir, "simulated_counts.npy"), simulated_counts)

    # evaluating the results
    samples = sampler.get_chain(flat=True)

    if save:
        # save pickled sampler
        filepath = os.path.join(save_dir, 'sampler.pkl')
        file = open(filepath, "wb")
        pickle.dump(sampler, file)
        file.close()

        # save flat chain
        filepath = os.path.join(save_dir, 'samples.npy')
        np.save(filepath, samples)

    # plot the OG
    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    simulator.plot_burst()

    # samples from posterior overlayed on true model
    plt.subplot(122)
    N_samples = 100
    plot_posterior_samples(N_samples, simulated_counts, samples, inf_params, modelparams)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "original_vs_posterior_samples.png")) if save else None

    # corner plot (or histogram for 1D)
    var_names = gen_parameter_labels(inf_params, N)
    if ndim > 1:
        fig = corner(samples, labels=var_names, truths=true_values)
    else:
        plt.figure()
        plot_1d_hist(samples, true_values)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"corner_plot.png")) if save else None

    plt.show() if show_plots else None

    # trace plot with arviz
    inference_data = az.from_emcee(sampler, var_names) 
    az.plot_trace(inference_data, compact=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"trace_plot.png")) if save else None

    plt.show() if show_plots else None

    # print some stats
    print(az.summary(inference_data))
    print("Autocorrelation time: ", sampler.get_autocorr_time())
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling algorithm")

    valid_FRBs = {'FRB20190501B_lc', 'FRB20190115B_lc'}
    noise_options = {"poisson", "gaussian"}

    parser.add_argument("-N","--ncomp", type=int, default=2, help="Number of burst components in sampled data")
    parser.add_argument("-i","--inf_params", nargs='+', help="List of inference parameter names.", required=True)
    parser.add_argument("--noise", type=str, help="Type of noise added to training data (poisson or gaussian).", required=True, choices=noise_options)

    parser.add_argument("-w", "--walkers", type=int, default=20,   help="Number of walkers. Default is 20.")
    parser.add_argument("--max_steps",   type=int, default=10_000, help="Maximum number of iterations before convergence. Default is 10000")
    parser.add_argument("-b", "--burn",    type=int, default=1500,  help="burn steps, number of burn-in steps used. Default is 1500.")
    parser.add_argument("--fast", action="store_true", help="speed up sampling by using parallelization")
    parser.add_argument("--save", action="store_true", help="save run to files and evaluation plots as png")
    parser.add_argument("--show_plots", action="store_true", help="show plots at the end of run")
    parser.add_argument("-f", "--FRB", type=str, default=None, choices=valid_FRBs, help="Name of FRB flux file to run mcmc on. True values are assumed unknown in this case.")

    args = parser.parse_args()

    main(args.ncomp, args.inf_params, args.noise, args.walkers, args.burn, args.max_steps,
        args.fast, args.save, args.show_plots, args.FRB
        )