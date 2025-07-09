import argparse
import os
import sys

from corner import corner

sys.path.append('..')

from src.simulator import Model, BurstSimulator
from src.flow_matching.distributions import Prior, Posterior, UniformPrior, CompositePrior, NewPosterior
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.training import GuidedConditionalFlowMatchingTrainer
from src.flow_matching.integration import EulerODESolver
from src.flow_matching.models import MLPGuidedVectorField, FRBLightCurveCNN, LightCurveThinner, LightCurveMLP, fourier_embedding
from src.flow_matching.transformer import TransformerGuidedField
from src.helpers import record_every, gen_parameter_labels

import numpy as np
from matplotlib import pyplot as plt
import torch

from src.flow_matching.plotting import plot_loss, plot_snapshots
from src.flow_matching.helpers import choose_device, build_mlp

def evaluation_plots(losses, vector_field, path, device, num_samples, inf_params, N, save_path, show_plots, 
                     loss=True, snapshots=True, corner_plot=True, 
                    ):

    # plot loss 
    if loss:
        plot_loss(losses, window_size=10, xlog=False, save_path=save_path, show=show_plots)
        plot_loss(losses, window_size=10, save_path=save_path, show=show_plots)

    if snapshots or corner_plot:
        # fixed model parameters (NOTE: Should be same as used during training in Posterior)
        # NOTE: these have to obey the prior! (i.e. t0_1 < t0_2 < ... < t0_n)
        time = np.linspace(0, 1.0, 1000)
        amp  = 100.0
        rise = 0.03
        skew = 5
        ybkg = 5.0
         
        burstparams = {
        't0'   : np.sort(np.random.rand(N)),
        'amp'  : [amp for _ in range(N)],
        'rise' : [rise for _ in range(N)],
        'skew' : [skew for _ in range(N)]
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

        final_snapshot = xts[:, -1, :]
    
    # plot snapshots of marginal path if space is 2D
    if snapshots and N * len(inf_params) == 2:
        plot_snapshots(xts, ts, record_every_idxs, num_marginals, save_path, show_plots)

    # corner plot
    if corner_plot:
        
        var_names = gen_parameter_labels(inf_params, N)
        true_values = np.array([simulator.get_true(key) for key in inf_params]).flatten()
        fig = corner(final_snapshot.cpu().numpy(), labels=var_names, truths=true_values)

        if save_path:
            filepath = os.path.join(save_path, 'corner_plot.png')
            fig.savefig(filepath, bbox_inches="tight") 

        plt.show() if show_plots else None

    # TODO: add plt of posterior samples over model and data 

def main(ncomp: int, inf_params: list[str],
        model: str, encoder: str, encode_tau: bool, encode_theta: bool, combine_mode:str,
        epochs: int, batch_size: int, lr: float, clip: int | None, EMA: bool,
        num_samples: int, show_plots: bool, no_save: bool, job_id: int | None
        ):
    """
    Train and evaluate flow matching model.
    """
    device = choose_device()
    print("CURRENT DEVICE: ", device)

    # TRAINING
    
    # number of burst components
    N = ncomp                

    print(f"\n Training data will have {N} burst components and sample {inf_params} from the prior \n")

    # define the priors
    PRIORS = {
        "t0"  : UniformPrior(x_min=0,    x_max=1,   log=False, enforce_order=True, dim=N),
        "amp" : UniformPrior(x_min=10,   x_max=300, log=False,  enforce_order=False, dim=N),
        "rise": UniformPrior(x_min=1e-3, x_max=1,  log=True, enforce_order=False, dim=N),
        "skew": UniformPrior(x_min=1,    x_max=6,   log=False, enforce_order=False, dim=N)
    }

    # only use priors of learnable parameters
    prior_dict = {param : PRIORS[param] for param in inf_params}

    # standard burst parameter values when fixed
    TIME = torch.linspace(0, 1.0, 1000)
    YBKG = 5.0
    AMP  = 100.0
    RISE = 0.03
    SKEW = 5
    
    burstparams = {
        't0'   : torch.sort(torch.rand(N))[0],
        'amp'  : torch.Tensor([AMP]).repeat(N),
        'rise' : torch.Tensor([RISE]).repeat(N),
        'skew' : torch.Tensor([SKEW]).repeat(N)
    }

    # inference parameters are not fixed
    for key in inf_params:
        burstparams[key] = None 

    modelparams = {
        'time' : TIME,
        'ncomp': N,
        'burstparams': burstparams,
        'ybkg': YBKG
    }

    prior     = CompositePrior(prior_dict)
    posterior = NewPosterior(modelparams, inf_params, prior)

    path = GuidedLinearProbabilityPath(
        p_simple = prior,
        p_data   = posterior
    )

    if encoder == "CNN":
        latent_dim = 128
        time_seq_encoder = FRBLightCurveCNN(latent_dim=latent_dim)
    elif encoder == "THIN":
        latent_dim = 100
        time_seq_encoder = LightCurveThinner(latent_dim=100)
    elif encoder == "MLP":
        latent_dim = 128
        time_seq_encoder = LightCurveMLP(layers=[1000, 512, 256, 128])
    else:
        latent_dim = 1000
        time_seq_encoder = None

    dim = N * len(inf_params)

    tau_encoder = None
    if encode_tau:
        tau_encoder=fourier_embedding

    theta_encoder = None
    if encode_theta and model != "T":
        theta_encoder = lambda theta_dim : build_mlp([dim] + [dim * 4, dim * 8] + [theta_dim])
    elif encode_theta:
        param_dim = len(inf_params)
        theta_encoder = lambda theta_dim : build_mlp([param_dim] + [param_dim * 8, param_dim * 32] + [theta_dim])

    if model == "MLP":
        vector_field = MLPGuidedVectorField(dim, [64, 64, 32, 16], latent_dim, time_seq_encoder, tau_encoder, theta_encoder, combine_mode)
    elif model == "T":
        vector_field = TransformerGuidedField(dim, inf_params, latent_dim, time_seq_encoder, tau_encoder, theta_encoder)
    
    trainer = GuidedConditionalFlowMatchingTrainer(path, vector_field)
    losses = trainer.train(epochs, device, lr, clip, EMA, False if no_save else True, batch_size, job_id)

    print('\n')

    #  EVALUATION

    # where to store the plots
    if no_save:
        save_path = None
    else:
        save_path = os.path.join(trainer.save_path, 'plots')
        os.makedirs(save_path, exist_ok=True)

    evaluation_plots(
        losses, vector_field, path, device, num_samples, 
        inf_params, N, save_path, show_plots)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model and save evaluation plots")
    
    # training data
    parser.add_argument("-N","--ncomp", type=int, default=2, help="Number of burst components in training data")
    parser.add_argument("-i","--inf_params", nargs='+', help="List of inference parameter names to learn from training data.", required=True)

    # ML model
    parser.add_argument("-m","--model", type=str, default="MLP", help="Model to train (MLP or T) T=Transformer")

    parser.add_argument("-c", "--encoder", type=str, default="MLP", help="Time series encoder (CNN or THIN or MLP or NULL)")
    parser.add_argument("--encode_tau", action="store_true", help="Use fourier embedding for tau (flow matching time)")
    parser.add_argument("--encode_theta", action="store_true", help="Use MLP embedding for tau (flow matching time)")
 
    parser.add_argument("--combine_mode", type=str, default="concat", help="how to combine the vectors [GLU or concat]")

    # training
    parser.add_argument("-e","--epochs", type=int, default=100_000, help="epochs")
    parser.add_argument("-b","--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-l", "--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--clip", type=float, default=None, help="Max norm of the gradient")
    parser.add_argument("--EMA", action="store_true", help="Use Exponential Model Averaging")

    # evaluation
    parser.add_argument("-s", "--num_samples", type=int, default=20_000, help="Number of samples used to construct final posterior")
    parser.add_argument("--show_plots", action="store_true", help="show evaluation plots in interactive window")
    parser.add_argument("--no_save", action="store_true", help="Do not save training checkpoints and plots (not recommended for long runs)")

    parser.add_argument("-j", "--job_id", type=int, default=None, help="job id when running on cluster (slurm)")

    args = parser.parse_args()

    valid_models = ["MLP", "T"]
    valid_encoders = ["CNN", "THIN", "MLP", "NULL"]
    valid_combine_modes = ["GLU", "concat"]
    valid_inf_params = {"t0", "amp", "skew", "rise"}

    # check correctness of args
    if args.model not in valid_models:
        print(f"model argument \'{args.model}\' invalid, must be in {valid_models}")
    elif args.encoder not in valid_encoders:
        print(f"encoder argument \'{args.encoder}\' invalid, must be in {valid_encoders}")
    elif args.combine_mode not in valid_combine_modes:
        print(f"Combine mode argument \'{args.combine_mode}\' invalid, must be in {valid_combine_modes}")
    elif len(set(args.inf_params) & valid_inf_params) != len(args.inf_params):
        print(f"inference parameters argument {args.inf_params} invalid, must be in {valid_inf_params}")
    else:
        main(
            args.ncomp, args.inf_params,
            args.model, args.encoder, args.encode_tau, args.encode_theta, args.combine_mode,
            args.epochs, args.batch_size, args.lr, args.clip, args.EMA, 
            args.num_samples, args.show_plots, args.no_save, args.job_id
            )
