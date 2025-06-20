import torch 
import matplotlib.pyplot as plt
import numpy as np

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

if __name__=="__main__":
    pass
