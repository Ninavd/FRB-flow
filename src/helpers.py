import numpy as np
import matplotlib.pyplot as plt
import torch 
import matplotlib.cm as cm

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

# Several plotting utility functions
def hist2d_samples(samples, bins: int = 200, percentile: int = 99, range=None, **kwargs):
    """
    Plots 2d histogram of given (x,y) samples. 

    Parameters:
      - samples (N, 2): samples to plot
      - bins (int): Number of bins
      - percentile (int): Used for color normalization
      - range (int): range of the histogram bins
    """
    if range is None:
        xmin, xmax = min(samples[:, 0]), max(samples[:, 0])
        ymin, ymax = min(samples[:, 1]), max(samples[:, 1])
        range = [[xmin, xmax], [ymin, ymax]]

    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=range)

    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(H.T, origin='lower', extent=extent, norm=norm, **kwargs)

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


if __name__=="__main__":
    pass
