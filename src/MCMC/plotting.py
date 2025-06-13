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
    - simulated_counts (ndarray[int]): Ground truth lightcurve
    - samples (ndarray): Parameter samples from posterior
    - inf_params (list[str]): labels of inferred parameters (t0, skew, amp, rise)
    - modelparams (dict): includes all parameters needed to simulate/model a lightcurve
    """
    time = modelparams["time"]
    plt.plot(time, simulated_counts, 'k-', alpha=0.1, label="data")

    for i in range(N):
        random_index = np.random.randint(low=0, high=len(samples))
        random_sample = samples[random_index]
        modelparams = update_modelparams(random_sample, inf_params, modelparams)
        model = Model(**modelparams).get_flux()
        plt.plot(time, model, alpha=0.5, label=f"{'posterior samples' if i == 0 else ''}", color='gray')

    plt.title(f'{N} posterior samples')
    plt.legend()