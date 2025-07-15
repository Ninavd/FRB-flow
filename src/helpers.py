from copy import deepcopy
import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Iterable

from src.simulator import Model

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the time indices corresponding to a `record_every` parameter.
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

def update_modelparams(sample, inf_params, modelparams):
    """
    Extract parameters from flat sample to update parameters dictionary.
    """
    N = modelparams['ncomp']
    for i, key in enumerate(inf_params):
        start = i * N
        stop  = start + N
        modelparams['burstparams'][key] = sample[start:stop]
    return modelparams

def plot_posterior_samples(N, simulated_counts, samples, inf_params, modelparams, true_flux=None, show=False, save_path=None):
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
    plt.plot(time, simulated_counts, 'k-', alpha=0.2, label="data")
    
    # prevents changing the original modelparams
    modelparams_copy = deepcopy(modelparams)

    for _ in range(N):

        # draw random sample from posterior
        random_index = np.random.randint(low=0, high=len(samples))
        random_sample = samples[random_index]

        # generate noise-free curve from random sample
        modelparams_copy = update_modelparams(random_sample, inf_params, modelparams_copy)
        model = Model(**modelparams_copy).get_flux()

        # plot the sample
        plt.plot(time, model, alpha=0.3)

    if true_flux is not None:
        plt.plot(np.linspace(0, 1, len(true_flux)), true_flux, 'r--', linewidth=1, label="ground-truth")
    
    plt.title(f'{N} posterior samples')
    plt.legend()
    plt.tight_layout()

    if save_path:
        filepath = os.path.join(save_path, 'posterior_samples.png')
        plt.savefig(filepath, bbox_inches="tight") 
    
    plt.show() if show else None

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

if __name__=="__main__":
    pass
