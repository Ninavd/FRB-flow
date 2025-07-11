import argparse
import os
import sys

sys.path.append('..')
from copy import deepcopy
from src.flow_matching.distributions import UniformPrior, CompositePrior, Posterior
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.training import GuidedConditionalFlowMatchingTrainer
from src.flow_matching.models import MLPGuidedVectorField, FRBLightCurveCNN, LightCurveThinner, LightCurveMLP, fourier_embedding
from src.flow_matching.transformer import TransformerGuidedField

import torch

from src.flow_matching.plotting import evaluation_plots
from src.flow_matching.helpers import choose_device, build_mlp

def pick_timeseries_encoder(type):
    """
    Select light curve encoder from given CLA.
    """
    if type == "CNN":
        latent_dim = 128
        time_seq_encoder = FRBLightCurveCNN(latent_dim=latent_dim)

    elif type == "THIN":
        latent_dim = 100
        time_seq_encoder = LightCurveThinner(latent_dim)

    elif type == "MLP":
        latent_dim = 128
        time_seq_encoder = LightCurveMLP(layers=[1000, 512, 256, 128])
        
    else:
        latent_dim = 1000
        time_seq_encoder = None

    return latent_dim, time_seq_encoder

def pick_theta_encoder(encode, inf_params, model, vector_dim):
    """
    Transformer: parameters of each burst component pass through independently.
    MLP: the entire sample is passed through the encoder.
    Hence different input dimensions are required for each one. 
    """
    if encode and model != "T":
        return lambda theta_dim : build_mlp([vector_dim, vector_dim * 4, vector_dim * 8, theta_dim])
    
    elif encode:
        param_dim = len(inf_params)
        return lambda theta_dim : build_mlp([param_dim] + [param_dim * 8, param_dim * 32] + [theta_dim])
    
    return None

def main(ncomp: int, inf_params: list[str], lambda_: list[float],
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
    lambda_ =  torch.repeat_interleave(torch.tensor(lambda_, device=device), repeats=N)   

    print(f"\n Training data will have {N} burst component{'' if N == 1 else 's'} and sample {inf_params} from the prior \n")

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
        't0'   : torch.linspace(0.1, 0.8, N),
        'amp'  : torch.Tensor([AMP]).repeat(N),
        'rise' : torch.Tensor([RISE]).repeat(N),
        'skew' : torch.Tensor([SKEW]).repeat(N)
    }

    modelparams = {
        'time' : TIME,
        'ncomp': N,
        'burstparams': burstparams,
        'ybkg': YBKG
    }

    prior     = CompositePrior(prior_dict)
    posterior = Posterior(deepcopy(modelparams), inf_params, prior)

    path = GuidedLinearProbabilityPath(
        p_simple = prior,
        p_data   = posterior
    )

    # dimension of vector field
    dim = N * len(inf_params)

    # encoders
    latent_dim, time_seq_encoder = pick_timeseries_encoder(encoder)
    tau_encoder   = fourier_embedding if encode_tau else None
    theta_encoder = pick_theta_encoder(encode_theta, inf_params, model, dim)
    
    if model == "MLP":
        vector_field = MLPGuidedVectorField(dim, [64, 64, 32, 16], latent_dim, time_seq_encoder, tau_encoder, theta_encoder, combine_mode)
    elif model == "T":
        vector_field = TransformerGuidedField(dim, inf_params, latent_dim, time_seq_encoder, tau_encoder, theta_encoder)
    
    trainer = GuidedConditionalFlowMatchingTrainer(path, vector_field)
    losses = trainer.train(epochs, device, lr, clip, EMA, False if no_save else True, batch_size, job_id, lambda_=lambda_)

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
        inf_params, N, modelparams, lambda_, save_path, show_plots
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model and save evaluation plots")
    
    # training data
    parser.add_argument("-N","--ncomp", type=int, default=2, help="Number of burst components in training data")
    parser.add_argument("-i","--inf_params", nargs='+', help="List of inference parameter names to learn from training data.", required=True)
    parser.add_argument("--lambda_", nargs='+', help="List of linear scaling vectors for inference parameter", type=float, required=True)

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
    elif len(args.inf_params) != len(args.lambda_):
        print(f"number of inference parameters must match number of scaling factors.")
    else:
        main(
            args.ncomp, args.inf_params, args.lambda_,
            args.model, args.encoder, args.encode_tau, args.encode_theta, args.combine_mode,
            args.epochs, args.batch_size, args.lr, args.clip, args.EMA, 
            args.num_samples, args.show_plots, args.no_save, args.job_id
            )
