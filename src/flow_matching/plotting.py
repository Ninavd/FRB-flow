import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import os

from numpy.lib.stride_tricks import sliding_window_view

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

def plot_loss(losses, window_size=None, ylog=True, xlog=True, save_path=None, show=True, **kwargs):
    """Plot evolution of loss over time."""

    # calculate running average
    if window_size:
        windows = sliding_window_view(losses, window_shape=window_size)
        running_avg = windows.mean(axis=1)

    plt.figure()
    plt.plot(losses, alpha=0.3, **kwargs)
    plt.plot(running_avg, label=f"running average N={window_size}") if window_size else None
    plt.yscale('log') if ylog else None
    plt.xscale('log') if xlog else None
    plt.xlabel('step')
    plt.ylabel("average loss")
    plt.title("Evolution of loss")
    plt.legend()

    if save_path:
        filename = f'loss_{"log" if ylog else "linear"}-{"log" if xlog else "linear"}.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, bbox_inches="tight")
    
    plt.show() if show else None

def plot_snapshots(xts, ts, record_every_idxs, num_marginals, save_path=None, show=True):
        """
        Plot snapshots of the marginal probability path.

        Args
            xts: (B, nts, dim) nts snapshots of each trajectory.
            ts: 
            record_every_idxs:
            num_marginals: number of snapshots to plot.
            save_path: path to saving directory.

        Returns
            xx: (B, 1, dim) Final snapshot
        """
        plt.figure(figsize=(20, 5))

        for idx in range(xts.shape[1]):
            xx = xts[:,idx,:]
            t = ts.cpu()[record_every_idxs[idx]]

            plt.subplot(1, num_marginals, idx + 1)
            
            hist2d_samples(xx.cpu(), range=[[0,1], [0,1]], percentile=100, cmap='Blues')
            
            plt.title(f"Learned, t={t:.2f}")
            plt.xlabel(f"t0_1")
            plt.ylabel(f"t0_2")

            plt.xlim(0, 1)
            plt.ylim(0, 1)

        if save_path:
            filepath = os.path.join(save_path, 'marginal_path_snapshots.png')
            plt.savefig(filepath, bbox_inches="tight")
        
        plt.show() if show else None

        return xx