import matplotlib.pyplot as plt
import numpy as np

from src.simulator import Model
from src.MCMC.helpers import update_modelparams

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

def plot_posterior_samples(N, simulated_counts, samples, inf_params, modelparams):
    """
    Overlay samples from posterior on simulated curve.

    Args:
    - N (int): Number of samples from posterior.
    - simulated_counts (ndarray[int]): Ground truth lightcurve.
    - samples (ndarray): Samples from posterior distribution of ground truth lightcurve.
    - inf_params (list[str]): Keys of inferred parameters (t0, skew, amp, rise).
    - modelparams (dict): All parameters needed to simulate/model a lightcurve.
    """
    # plot the data
    time = modelparams["time"]
    plt.plot(time, simulated_counts, 'k-', alpha=0.1, label="data")

    for _ in range(N):

        # draw random sample from posterior
        random_index = np.random.randint(low=0, high=len(samples))
        random_sample = samples[random_index]

        # generate noise-free curve from random sample
        modelparams = update_modelparams(random_sample, inf_params, modelparams)
        model = Model(**modelparams).get_flux()

        # plot the sample
        plt.plot(time, model, alpha=0.3)

    plt.title(f'{N} posterior samples')
    plt.legend()


def plot_trace(sampler, steps, true_value, ylabel="$t_0$"):
    nwalkers = sampler.nwalkers

    plt.figure(figsize=(12,3))

    # each walker a different color
    plt.subplot(121)
    for i, walker in enumerate(sampler.get_chain().reshape((steps, nwalkers)).T):
        start = int(i * steps)
        plt.plot(range(start, start + steps), walker)

    plt.title("MCMC traceplot")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.hlines([true_value], xmin=0, xmax=nwalkers*steps, linestyle='dashed', color='black')

    # walkers back to back
    plt.subplot(122)
    plt.plot(sampler.get_chain(flat=True), 'k-')
    plt.title("MCMC traceplot")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.hlines([true_value], xmin=0, xmax=nwalkers*steps, linestyle='dashed', color='red')