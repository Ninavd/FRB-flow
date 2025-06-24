import argparse
import os
import sys

from corner import corner

sys.path.append('..')

from src.simulator import Model, BurstSimulator
from src.flow_matching.distributions import Prior, Posterior
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.training import GuidedConditionalFlowMatchingTrainer
from src.flow_matching.integration import EulerODESolver
from src.flow_matching.models import MLPGuidedVectorField, FRBLightCurveCNN, LightCurveThinner, fourier_embedding
from src.helpers import record_every

import numpy as np
from matplotlib import pyplot as plt
import torch

from src.flow_matching.plotting import plot_loss, plot_snapshots
from src.flow_matching.helpers import choose_device, build_mlp

def evaluation_plots(losses,  vector_field, path, device, num_samples, save_path, show_plots, 
                     loss=True, snapshots=True, corner_plot=True
                    ):

    # plot loss 
    if loss:
        plot_loss(losses, window_size=10, xlog=False, save_path=save_path, show=show_plots)
        plot_loss(losses, window_size=10, save_path=save_path, show=show_plots)

    # fixed model parameters (NOTE: Should be same as used during training in Posterior)
    if snapshots or corner_plot:
        N = 2
        time = np.linspace(0, 1.0, 1000)
        amp  = 100.0
        rise = 0.03
        skew = 5
        ybkg = 5.0
        true_t0 = [0.2, 0.8] # NOTE: this has to obey the prior! (i.e. t0_1 < t0_2 < ... < t0_n)

        # generate simulated data for one burst
        burstparams = {
            't0'   : true_t0, 
            'amp'  : [amp, amp],
            'rise' : [rise, rise],
            'skew' : [skew, skew]
        }

        # generate one instance of simulated data to guide prior samples with
        model = Model(time=time, ncomp=N, burstparams=burstparams, ybkg=ybkg)
        simulator = BurstSimulator(model)
        x_counts = simulator.simulate_burst() 

        num_samples = num_samples  # number of prior samples to transform 
        num_marginals = 5   # number of snapshots

        # TODO: maybe do this in batches
        # use same data point for conditioning all prior samples
        simulations = torch.tensor(x_counts, device=device, dtype=torch.float).repeat(num_samples, 1)

        # initialize ODE solver
        solver = EulerODESolver(vector_field)
        nts = 100
        ts = torch.linspace(0, 1, nts).to(device)

        # simulate ODE starting from x0
        x0 = path.p_simple.sample(num_samples).to(device)
        xts = solver.solve_with_trajectory(x0, ts.view(1, nts, 1).expand(num_samples, nts, 1), y=simulations)

        # only save num_marginals snapshots 
        record_every_idxs = record_every(nts, nts // (num_marginals - 1))
        xts = xts[:, record_every_idxs, :]

        # plot snapshots of marginal path
        final_snapshot = plot_snapshots(xts, ts, record_every_idxs, num_marginals, save_path, show_plots)

    # corner plot
    if corner_plot:
        fig = corner(final_snapshot.cpu().numpy(), labels=["t0_1", "t0_2"], truths=np.array(true_t0))

        if save_path:
            filepath = os.path.join(save_path, 'corner_plot.png')
            fig.savefig(filepath, bbox_inches="tight") 

        plt.show() if show_plots else None

def main(model: str, encoder: str, encode_tau: bool, encode_theta: bool, combine_mode:str,
         epochs: int, batch_size: int, lr: float, num_samples:int, 
         show_plots: bool, no_save: bool, job_id: int | None
         ):
    """
    Train and evaluate flow matching model.
    """
    device = choose_device()
    print("CURRENT DEVICE: ", device)

    # TRAINING

    path = GuidedLinearProbabilityPath(
        p_simple=Prior(),
        p_data=Posterior()
    )

    if encoder == "CNN":
        latent_dim = 128
        time_seq_encoder = FRBLightCurveCNN(latent_dim=latent_dim)
    elif encoder == "THIN":
        latent_dim = 100
        time_seq_encoder = LightCurveThinner(latent_dim=100)
    else:
        latent_dim = 1000
        time_seq_encoder = None

    dim = 2

    tau_encoder = None
    if encode_tau:
        tau_encoder=fourier_embedding

    theta_encoder = None
    if encode_theta:
        theta_encoder = lambda theta_dim : build_mlp([dim] + [dim * 4, dim * 8] + [theta_dim])

    if model == "MLP":
        vector_field = MLPGuidedVectorField(dim=2, hiddens=[64, 64, 32, 16], time_dim=latent_dim, 
                                            time_seq_encoder=time_seq_encoder, tau_encoder=tau_encoder, theta_encoder=theta_encoder, 
                                            combine=combine_mode
                                            )
    
    trainer = GuidedConditionalFlowMatchingTrainer(path, vector_field)
    losses = trainer.train(epochs, device, lr, False if no_save else True, batch_size, job_id)

    print('\n')

    #  EVALUATION

    # where to store the plots
    if no_save:
        save_path = None
    else:
        save_path = os.path.join(trainer.save_path, 'plots')
        os.makedirs(save_path, exist_ok=True)

    evaluation_plots(losses, vector_field, path, device, num_samples, save_path, show_plots)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model and save evaluation plots")

    parser.add_argument("-m","--model", type=str, default="MLP", help="Model to train")

    parser.add_argument("-c", "--encoder", type=str, default="CNN", help="Time series encoder (CNN or THIN or NULL)")
    parser.add_argument("--encode_tau", action="store_true", help="Use fourier embedding for tau (flow matching time)")
    parser.add_argument("--encode_theta", action="store_true", help="Use MLP embedding for tau (flow matching time)")
 
    parser.add_argument("--combine_mode", type=str, default="concat", help="how to combine the vectors [GLU or concat]")

    parser.add_argument("-e","--epochs", type=int, default=100_000, help="epochs")
    parser.add_argument("-b","--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-l", "--lr", type=float, default=5e-4, help="learning rate")

    parser.add_argument("-s", "--num_samples", type=int, default=20_000, help="Number of samples used to construct final posterior")
    parser.add_argument("--show_plots", action="store_true", help="show evaluation plots in interactive window")
    parser.add_argument("--no_save", action="store_true", help="Do not save training checkpoints and plots (not recommended for long runs)")

    parser.add_argument("-j", "--job_id", type=int, default=None, help="job id when running on cluster (slurm)")

    args = parser.parse_args()

    valid_models = ["MLP"]
    valid_encoders = ["CNN", "THIN", "NULL"]
    valid_combine_modes = ["GLU", "concat"]

    # check correctness of args
    if args.model not in valid_models:
        print(f"model argument invalid, must be in {valid_models}")
    elif args.encoder not in valid_encoders:
        print(f"encoder argument invalid, must be in {valid_encoders}")
    elif args.combine_mode not in valid_combine_modes:
        print(f"Combine mode argument invalid, must be in {valid_combine_modes}")
    else:
        main(
            args.model, args.encoder, args.encode_tau, args.encode_theta, args.combine_mode,
            args.epochs, args.batch_size, args.lr, args.num_samples, args.show_plots, args.no_save, args.job_id)
