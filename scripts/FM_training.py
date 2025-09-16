import argparse
import os
import sys

sys.path.append('..')
from copy import deepcopy
from datetime import datetime
from src.flow_matching.distributions import UniformPrior, CompositePrior, Posterior, DiscreteUniform
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.training import GuidedConditionalFlowMatchingTrainer, TransdimensionalTrainer
from src.flow_matching.models import (
    MLPGuidedVectorField,
    FRBLightCurveCNN,
    LightCurveThinner,
    LightCurveMLP,
    fourier_embedding,
    UNetEncoder,
    EncodedClassifier,
    TransdimensionalModel
)
from src.flow_matching.transformer import TransformerGuidedField

import numpy as np
import torch
import wandb

from src.flow_matching.plotting import evaluation_plots
from src.flow_matching.helpers import choose_device, build_mlp, get_sample_mean_std


def pick_timeseries_encoder(type: str) -> tuple[int, torch.nn.Module | None]:
    """Select light curve encoder from given CLA.

    Args:
        type (str): Encoder label.

    Returns:
        tuple: A tuple containing
            - int: Latent dimension of encoded time series.
            - nn.Module or None: The selected time series encoder.
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

    elif type == "UNET":
        latent_dim = 128
        time_seq_encoder = UNetEncoder(latent_dim=latent_dim)

    else:
        latent_dim = 1000
        time_seq_encoder = None

    return latent_dim, time_seq_encoder


def pick_theta_encoder(encode: bool, inf_params: list[str], model: str, dim: int) -> torch.nn.Module | None:
    """Selects compatible encoder for parameter vector theta.

    In a transformer the parameters of each burst component pass through independently, 
    while for MLPs the entire sample is passed through the encoder. Hence different 
    input dimensions are required for each one.

    Args:
        encode (bool): Whether theta is encoded.
        inf_params (list of str): List of inference parameter names.
        model (str): Model architecture name.
        dim (int): Dimension of the parameter vector.

    Returns:
        nn.Module or None: The selected theta encoder.
    """
    if encode and model != "T":
        return lambda theta_dim: build_mlp([dim, dim * 4, dim * 8, theta_dim])

    elif encode:
        param_dim = len(inf_params)
        return lambda theta_dim: build_mlp([param_dim] + [param_dim * 8, param_dim * 32] + [theta_dim])

    return None


def main(N: int,
         inf_params: list[str],
         noise: str,
         ybkg: float,
         model: str,
         encoder: str,
         encode_tau: bool,
         encode_theta: bool,
         combine_mode: str,
         add_pos_enc: bool,
         fixed_N:bool,
         epochs: int, batch_size: int, lr: float, clip: int | None, EMA: bool,
         num_samples: int, show_plots: bool, no_save: bool, job_id: int | None
         ):
    """Train and evaluate flow matching model.

    Args:
        N (int):                 Number of components.
        inf_params (list of str):List of inference parameter names.
        noise (str):             Type of noise added to flux (poisson or gaussian).
        ybkg (float):            Background rate added to flux.
        model (str):             Model architecture name.
        encoder (str):           Time series encoder type.
        encode_tau (bool):       Whether to encode tau.
        encode_theta (bool):     Whether to encode theta.
        combine_mode (str):      Method for injecting conditions.
        add_pos_enc (bool):      Use positional encoding in transformer. Auto-selects sorted t0 prior.
        epochs (int):            Number of training epochs.
        batch_size (int):        Size of each training batch.
        lr (float):              Learning rate.
        clip (int or None):      Gradient clipping threshold. 
        EMA (bool):              Use exponential moving average.
        num_samples (int):       Number of samples to draw during inference.
        show_plots (bool):       Display plots after training.
        no_save (bool):          Prevents model checkpoints and plots from being saved.
        job_id (int or None):    Job ID for logging purposes.
    """
    # =========================
    #       INITIALISATION
    # =========================
    
    device = choose_device()

    # define the priors
    enforce_order = True if add_pos_enc else False

    PRIORS = {
        "t0"  : UniformPrior(x_min=0.2,  x_max=0.8, log=False, enforce_order=enforce_order, dim=N, device=device),
        "amp" : UniformPrior(x_min=1,   x_max=300, log=True,  enforce_order=False, dim=N, device=device),
        "rise": UniformPrior(x_min=1e-3, x_max=0.1, log=True, enforce_order=False, dim=N, device=device),
        "skew": UniformPrior(x_min=1,    x_max=6,   log=False, enforce_order=False, dim=N, device=device)
    }

    # only use priors of learnable parameters
    PRIOR_DICT = {param : PRIORS[param] for param in inf_params}

    # standard burst parameter values when fixed
    TIME = torch.linspace(0, 1.0, 1000)
    YBKG = ybkg
    AMP  = 100
    RISE = 0.03
    SKEW = 5

    BURSTPARAMS = {
        't0'   : torch.linspace(0.2, 0.8, N),
        'amp'  : torch.Tensor([np.log10(AMP)]).repeat(N),
        'rise' : torch.Tensor([np.log10(RISE)]).repeat(N),
        'skew' : torch.Tensor([SKEW]).repeat(N)
    }

    MODELPARAMS = {
        'time' : TIME,
        'ncomp': N,
        'burstparams': BURSTPARAMS,
        'ybkg': YBKG
    }
    N_min = 1 if not fixed_N else N 

    prior     = CompositePrior(PRIOR_DICT, device=device)
    N_prior   = DiscreteUniform(N_min, N, device=device)
    posterior = Posterior(deepcopy(MODELPARAMS), inf_params, prior, N_prior, noise)

    path = GuidedLinearProbabilityPath(
        p_simple = prior,
        p_data   = posterior
    )

    N_prior_samples = N_prior.sample(10_000)
    MEAN, STD = get_sample_mean_std(prior, N, inf_params, N_prior_samples, num_samples=10_000, device=device)    

    # dimension of vector field
    DIM = N * len(inf_params)

    # encoders
    tau_encoder   = fourier_embedding if encode_tau else None
    theta_encoder = pick_theta_encoder(encode_theta, inf_params, model, DIM)
    latent_dim, time_seq_encoder = pick_timeseries_encoder(encoder)
    
    if model == "MLP":
        vector_field = MLPGuidedVectorField(
             DIM, [64, 64, 32, 16],
             latent_dim,
             time_seq_encoder,
             tau_encoder,
             theta_encoder,
             combine_mode
            )

    elif model == "T":
        vector_field = TransformerGuidedField(
                         DIM,
                         inf_params,
                         latent_dim,
                         time_seq_encoder,
                         tau_encoder,
                         theta_encoder,
                         add_pos_encoding=add_pos_enc
                         )

    if fixed_N:
        trainer = GuidedConditionalFlowMatchingTrainer(path, vector_field)
    else:
        time_dim, classifier_time_encoder = pick_timeseries_encoder(encoder)
        N_classifier = EncodedClassifier(
                         classifier_time_encoder,
                         inputs=time_dim,
                         hiddens=[time_dim // 2, time_dim // 2, time_dim // 4, time_dim // 4],
                         outputs=N
                         ).to(device)
        vector_field = TransdimensionalModel(N_classifier, vector_field)
        trainer = TransdimensionalTrainer(path, vector_field)

    # ==============================
    #           TRAINING
    # ==============================

    print("CURRENT DEVICE: ", device)
    print(
        f"""
        \n Training data will have {'at maximum ' if not fixed_N else ''} {N} burst component{'' if N == 1 else 's'}
        and sample {inf_params} from the prior\n
        """
    )

    # start a new wandb run to track this script
    if not no_save:
        run = wandb.init(
            entity = "ninavd-university-of-amsterdam",
            project= "frb-flow",
            config = trainer.make_config(epochs, lr, clip, batch_size, path, fixed_N, MEAN, STD),
            name   = f"{job_id if job_id else ''}_" + datetime.now().strftime("%d_%m_%H%M")
        )

    losses = trainer.train(
         epochs,
         device,
         lr,
         clip,
         EMA,
         False if no_save else True,
         batch_size,
         job_id,
         fixed_N=fixed_N,
         mean=MEAN, std=STD
         )

    print('\n')



    # ============================
    #         EVALUATION     
    # ============================

    # where to store the plots
    if no_save:
        save_path = None
    else:
        save_path = os.path.join(trainer.save_path, 'plots')
        os.makedirs(save_path, exist_ok=True)

    evaluation_plots(
        losses, vector_field, path, device, num_samples, 
        inf_params, N, MODELPARAMS, MEAN, STD, save_path, show_plots
    )

    # save eval plots in wandb
    if not no_save:

        for filename in os.listdir(save_path):
            if 'loss' not in filename:
                filepath = os.path.join(save_path, filename)
                wandb.log({filename: wandb.Image(filepath)})
    
    wandb.finish() if not no_save else None

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model and save evaluation plots")
    
    valid_models = ["MLP", "T"]
    valid_encoders = ["CNN", "THIN", "MLP", "NULL", "UNET"]
    valid_combine_modes = ["GLU", "concat"]
    valid_inf_params = {"t0", "amp", "skew", "rise"}
    noise_options = ["gaussian", "poisson"]

    # training data
    parser.add_argument("-N", "--ncomp", type=int, default=2, help="Number of burst components in training data")
    parser.add_argument("-i", "--inf_params", nargs='+', help="List of inference parameter names to learn from training data.", required=True, choices=valid_inf_params)
    parser.add_argument("--noise", type=str, help="Type of noise added to training data (poisson or gaussian).", required=True, choices=noise_options)
    parser.add_argument("--ybkg", type=float, help="Background rate in training data. Default is 5", default=5)

    # ML model
    parser.add_argument("-m", "--model", type=str, default="MLP", help="Model to train (MLP or T) T=Transformer", choices=valid_models)

    parser.add_argument("-c", "--encoder", type=str, default="MLP", help="Time series encoder (CNN or THIN or MLP or T or UNET or NULL)", choices=valid_encoders)
    parser.add_argument("--encode_tau", action="store_true", help="Use fourier embedding for tau (flow matching time)")
    parser.add_argument("--encode_theta", action="store_true", help="Use MLP embedding for theta (parameter vector)")
 
    parser.add_argument("--combine_mode", type=str, default="concat", help="how to combine the vectors [GLU or concat]", choices=valid_combine_modes)
    parser.add_argument("--add_pos_enc", action="store_true", help="When selected model is transformer, use positional encoding. This also auto-selects a sorted t0 prior.")

    # training
    parser.add_argument("--fixed_N", action="store_true", help="Train on data with fixed number of burst components.")
    parser.add_argument("-e", "--epochs", type=int, default=100_000, help="epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-l", "--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--clip", type=float, default=None, help="Max norm of the gradient")
    parser.add_argument("--EMA", action="store_true", help="Use Exponential Model Averaging")

    # evaluation
    parser.add_argument("-s", "--num_samples", type=int, default=20_000, help="Number of samples used to construct final posterior")
    parser.add_argument("--show_plots", action="store_true", help="show evaluation plots in interactive window")
    parser.add_argument("--no_save", action="store_true", help="Do not save training checkpoints and plots (not recommended for long runs)")

    parser.add_argument("-j", "--job_id", type=int, default=None, help="job id when running on cluster (slurm)")

    args = parser.parse_args()

    # check correctness of args
    if len(set(args.inf_params) & valid_inf_params) != len(args.inf_params):
        print(f"inference parameters argument {args.inf_params} invalid, must be in {valid_inf_params} and contain no doubles")
    else:
        main(
            args.ncomp, args.inf_params, args.noise, args.ybkg,
            args.model, args.encoder, args.encode_tau, args.encode_theta, args.combine_mode, args.add_pos_enc,
            args.fixed_N, args.epochs, args.batch_size, args.lr, args.clip, args.EMA, 
            args.num_samples, args.show_plots, args.no_save, args.job_id
            )
