import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import os

from corner import corner
from numpy.lib.stride_tricks import sliding_window_view
from typing import Iterable, Any 

from src.simulator import Model, BurstSimulator
from src.flow_matching.integration import EulerODESolver
from src.flow_matching.models import TransdimensionalModel
from src.flow_matching.transformer import TransformerGuidedField

from src.helpers import record_every, plot_posterior_samples

import numpy as np
import torch

from src.helpers import gen_parameter_labels

def hist2d_samples(
        samples: np.ndarray, 
        bins: int = 200, 
        percentile: int = 99, 
        range:list[float]=None, 
        **kwargs
        ):
    """Plots 2d histogram of given (x,y) samples. 

    Args:
        samples (ndarray): samples of shape (N, 2).
        bins (int): Number of bins. Default is 200.
        percentile (int): Used for color normalization.
        range (list of float): range of the histogram bins. If None is calculated using min and max of samples.
        **kwargs: Keyword args to pass to `plt.imshow`
    
    Returns:
        list of int: bin range of the histogram
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
    aspect = max(range[0]) / max(range[1]) # square plot
    plt.imshow(H.T, origin='lower', extent=extent, norm=norm, aspect=aspect, **kwargs)
    return range

def plot_loss(
        losses: np.ndarray[float], 
        window_size: int|None =None, 
        ylog: bool=True, 
        xlog: bool=True, 
        save_path: str=None, 
        show: bool=True, 
        **kwargs
        ):
    """Plot evolution of loss over time.
    
    Args:
        losses (array of floats): loss at each step.
        window_size (int): Used for calculating running average.
        ylog (bool): set yscale to log. Default is `True`.
        xlog (bool): set xscale to log. Default is `True`.
        save_path (str or None): Path to saving directory. Default is None (no saving).
        show (bool): Show plot with `plt.show`. Default is True. 
        **kwargs: Keyword arguments passed to `plt.plot`
    """

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

def plot_snapshots(
        xts: torch.Tensor, 
        ts: torch.Tensor, 
        record_every_idxs: torch.Tensor, 
        num_marginals: int, 
        inf_params: list[str], 
        N: int, 
        save_path: str | None=None, 
        show: bool=True
        ) -> torch.Tensor:
    """
    Plot snapshots of the marginal probability path.

    Args
        xts (torch.Tensor): Tensor of shape (B, nts, dim) with snapshots of B trajectories.
        ts (torch.Tensor): Tensor of shape (nts,) with timestamps corresponding to snapshots.
        record_every_idxs (torch.Tensor): (num_marginals,) Time indices of snapshots to be plot.
        num_marginals (int): Number of snapshots to plot.
        inf_params (list of str): Inference parameter names. 
        N (int): Number of burst components. 
        save_path (str or None): Path to saving directory. The plot is not saved if None.
        show (bool): Show plot with `plt.show`.

    Returns
        torch.Tensor: Final snapshot of shape (B, 1, dim) 
    """
    plt.figure(figsize=(15, 4))

    labels = gen_parameter_labels(inf_params, N)

    for idx in range(xts.shape[1]):
        xx = xts[:,idx,:]
        t = ts.cpu()[record_every_idxs[idx]]

        plt.subplot(1, num_marginals, idx + 1)
        
        range_ = None if idx == 0 else range_
        range_ = hist2d_samples(xx.cpu(), range=range_, percentile=100, cmap='Blues')
        
        plt.title(f"Learned, t={t:.2f}")
        plt.xlabel(labels[0])
        plt.ylabel(labels[1]) if idx == 0 else None

    if save_path:
        filepath = os.path.join(save_path, 'marginal_path_snapshots.png')
        plt.savefig(filepath, bbox_inches="tight")
    
    plt.tight_layout()
    plt.show() if show else None

    return xx

def corner_plot(
        samples: np.ndarray, 
        burstparams: dict[str, Iterable], 
        inf_params: list[str], 
        N: int, 
        save_path: str | None, 
        show: bool, 
        **kwargs
        ):
    """
    Plot corner plot from posterior samples.

    Args
        samples (ndarray): Array with posterior samples of shape (B, dim).
        ts (torch.Tensor): Tensor of shape (nts,) with timestamps corresponding to snapshots.
        burstparams (dict[str, Iterable]): Parameters and their true value for each burst component.
        inf_params (list of str): Inference parameter names. 
        N (int): Number of burst components. 
        save_path (str or None): Path to saving directory. If None, the plot is not saved.
        show (bool): Show plot with `plt.show()`
        **kwargs: Arguments to pass to corner function.

    Returns
        torch.Tensor: Final snapshot of shape (B, 1, dim) 
    """
    var_names = gen_parameter_labels(inf_params, N)
    true_values = np.array([burstparams[key] for key in inf_params]).flatten()
    fig = corner(samples, labels=var_names, truths=true_values, **kwargs)

    if save_path:
        filepath = os.path.join(save_path, 'corner_plot.png')
        fig.savefig(filepath, bbox_inches="tight") 

    plt.show() if show else None

def evaluation_plots(
        losses: np.ndarray[float], 
        vector_field: torch.nn.Module, 
        path: torch.nn.Module, 
        device: torch.device, 
        num_samples: int, 
        inf_params: list[str], 
        N: int, 
        modelparams: dict[str, Any], 
        mean: torch.Tensor, 
        std: torch.Tensor,
        save_path: str|None, 
        show_plots: bool, 
        loss: bool=True, snapshots: bool=True, make_corner: bool=True, 
        ):
    """Evaluation plots made after training.
    
    Args:
        losses (array of floats): loss at each step.
        vector_field (nn.Module): Trained flow matching model. 
        path (nn.Module): Probability path used in training. 
        device (torch.device): Evaluate model on this device.
        num_samples (int): Number of samples used at inference.
        inf_params (list of str): Inference parameter names. 
        N (int): Number of burst components.
        modelparams (dict): Dictionary with kwargs for `Model`, the burst model. 
            The learned vector field is evaluated on a burst with these parameters.
        mean (torch.Tensor): Tensor of shape (), estimated mean of the prior. 
            Used for standardization of prior samples.
        std (torch.Tensor): Tensor of shape (), estimated standard deviation of the prior. 
            Used for standardization of prior samples.
        save_path (str or None), 
        show_plots (bool): show plots with `plt.show` 
        loss (bool): Whether to plot the loss. 
        snapshots (bool): Whether to plot the snapshots. Only works if the vector field is 2D. 
        make_corner (bool): Whether to plot the corner plot. 
    """
    # plot loss 
    if loss:
        plot_loss(losses, window_size=10, xlog=False, save_path=save_path, show=False)
        plot_loss(losses, window_size=10, save_path=save_path, show=show_plots)

    if snapshots or make_corner:
       # generate one instance of simulated data to guide prior samples with
        model = Model(**modelparams)
        simulator = BurstSimulator(model)
        x_counts = simulator.simulate_burst() 
        
        # we condition all prior samples on same simulation
        AVG_AMP = 150
        simulations = torch.tensor(x_counts / AVG_AMP, device=device, dtype=torch.float).repeat(num_samples, 1)
        
        # number of snapshots to plot
        num_marginals = 5   

        # TODO: maybe integrate in batches
        kwargs = {}
        if isinstance(vector_field, TransdimensionalModel):
            logits = vector_field.component_classifier(simulations)
            N_samples = torch.multinomial(torch.softmax(logits, dim=1), num_samples=1) + 1
            kwargs['N'] = N_samples
        elif isinstance(vector_field, TransformerGuidedField):
            kwargs['N'] = torch.tensor(N, device=device).repeat(num_samples, 1)

        # initialize ODE solver
        solver = EulerODESolver(vector_field)
        nts = 100
        ts = torch.linspace(0, 1, nts).to(device)

        # simulate ODE starting from x0
        x0 = (path.p_simple.sample(num_samples).to(device) - mean) / std

        # plot snapshots of marginal path if space is 2D
        if snapshots and N * len(inf_params) == 2:
            xts = solver.solve_with_trajectory(x0, ts.view(1, nts, 1).expand(num_samples, nts, 1), y=simulations, **kwargs)
            xts *= std + mean
            
            # only save num_marginals snapshots 
            record_every_idxs = record_every(nts, nts // (num_marginals - 1))
            xts = xts[:, record_every_idxs, :]

            final_snapshot = plot_snapshots(xts, ts, record_every_idxs, num_marginals, inf_params, N, save_path, show_plots)
        else:
            final_snapshot = solver.solve(x0, ts.view(1, nts, 1).expand(num_samples, nts, 1), y=simulations, **kwargs) * std + mean

    # corner plot
    if make_corner:
        corner_plot(final_snapshot.cpu().numpy(), modelparams['burstparams'], inf_params, N, save_path, show_plots)

    # plot posterior samples over simulated lightcurve 
    true_flux = model.get_flux()
    plt.figure()
    plot_posterior_samples(100, x_counts, final_snapshot.cpu().numpy(), inf_params, modelparams, true_flux, show_plots, save_path)
