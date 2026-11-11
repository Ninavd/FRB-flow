import os

import torch
import yaml

from src.flow_matching.distributions import (
    CompositePrior,
    DiscreteUniform,
    Posterior,
    StandardGaussian,
    UniformPrior,
)
from src.flow_matching.helpers import build_mlp, find_run_dir
from src.flow_matching.models import (
    EncodedClassifier,
    LightCurveMLP,
    MLPGuidedVectorField,
    TransdimensionalModel,
    UNetEncoder,
    fourier_embedding,
)
from src.flow_matching.probability_path import (
    ConditionalProbabilityPath,
    GuidedLinearProbabilityPath,
)
from src.flow_matching.transformer import TransformerGuidedField

"""
Functions used to manage saved runs, i.e.
load a trained model from conifg, or 
check if a run exists.
"""

# =====================
#       REGISTRY 
# =====================
MODELS = {
    "MLPGuidedVectorField": MLPGuidedVectorField,
    "TransformerGuidedField":TransformerGuidedField
}

TIME_ENCODERS = {
    "LightCurveMLP":LightCurveMLP,
    "UNetEncoder":UNetEncoder
}

TAU_ENCODERS = {
    True : fourier_embedding,
    False : None 
}

param_dim = 4
THETA_ENCODERS = {
    True : lambda theta_dim: build_mlp([param_dim] + [param_dim * 4, param_dim * 8, param_dim * 8, param_dim * 16, param_dim * 16] + [theta_dim]),
    # True : lambda theta_dim : build_mlp([4] + [4 * 8, 4 * 32] + [theta_dim]),
    False : None
}

PATHS = {
    "GuidedLinearProbabilityPath": GuidedLinearProbabilityPath,
}

DISTRIBUTIONS = {
    "Posterior":     Posterior,
    "UniformPrior":  UniformPrior,
    "CompositePrior":CompositePrior,
    "StandardGaussian": StandardGaussian
}

CLASSIFIERS = {
    "EncodedClassifier":EncodedClassifier
}

# ============================================


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def empty_model_from_config(config):
    """Creates empty flow matching model with args from the config.
    """
    model_name = config["model"]["name"]
    model_class = MODELS[model_name]
    
    kwargs = config["model"]["init_params"]

    time_encoder = False if not config["time_seq_encoder"] else config["time_seq_encoder"]["name"] 

    if time_encoder:
        t_encoder_kwargs = config["time_seq_encoder"]["init_params"]

    kwargs['time_seq_encoder']  = None if not time_encoder else TIME_ENCODERS[time_encoder](**t_encoder_kwargs)
    kwargs['tau_encoder']   = TAU_ENCODERS[config["tau_encoder"]]
    kwargs['theta_encoder'] = THETA_ENCODERS[config["theta_encoder"]]

    if not config["model"]["init_params"].get("encoder_kwargs"):
        print('NOTE: default arguments used for encoder layer')
    return model_class(**kwargs)

def empty_classifier_from_config(config):
    """Creates empty classifier model with args from the config.
    """
    classifier_conf = config.get("classifier", False)
    if not classifier_conf:
        print("No classifier settings found in config")
        return False
    
    classifier_name = classifier_conf["name"]
    classifier = CLASSIFIERS[classifier_name]

    encoder = TIME_ENCODERS[classifier_conf['encoder']['name']]
    enc_kwargs = classifier_conf['encoder']['init_params']
    encoder = encoder(**enc_kwargs)
    
    cls_kwargs = classifier_conf["init_params"]
    classifier = classifier(encoder, **cls_kwargs)

    return classifier


# loading the probability path
def prob_path_from_config(config):
    """
    Loads probability path from the config.
    """
    path_name = config["path"]["name"]
    path_class = PATHS[path_name]
    
    inf_params = config["path"]["p_data"]["init_params"]['inf_params']
    p_simple_name = config["path"]["p_simple"]["name"]
    p_simple_cls = DISTRIBUTIONS[p_simple_name]
    kwargs = config["path"]["p_simple"]["init_params"]
    
    if p_simple_name == "CompositePrior":
        prior_dict = {}

        for param, prior in zip(inf_params, kwargs):
            prior_cls = DISTRIBUTIONS[prior['name']]
            prior_dict[param] = prior_cls(**prior['init_params'])

        p_simple = p_simple_cls(prior_dict)

    else:
        p_simple = p_simple_cls(**kwargs)
    
    p_data_name   = config["path"]["p_data"]["name"]
    p_data_cls = DISTRIBUTIONS[p_data_name]

    kwargs = config["path"]["p_data"]["init_params"]
    kwargs['model_params']['time'] = torch.linspace(0, 1, 1000)
    
    N_prior = config['training'].get('fixed_N', None)
    if N_prior:
        N_prior = DiscreteUniform(kwargs['model_params']['ncomp'], kwargs['model_params']['ncomp'])
    else:
        N_prior = None

    # for old version
    if p_simple_name == "CompositePrior":
        kwargs.pop('prior') if kwargs.get('prior', False) else None
        p_data = p_data_cls(prior=p_simple, N_prior=N_prior, **kwargs)
    # for new version
    else:
        # print(kwargs)
        prior_cls = DISTRIBUTIONS[kwargs['prior']['name']]
        prior_dict = {}
        for param, sub_prior in zip(inf_params, kwargs['prior']['init_params']):
            sub_prior_cls = DISTRIBUTIONS[sub_prior['name']]
            # print(sub_prior['init_params'])
            prior_dict[param] = sub_prior_cls(**sub_prior['init_params'])

        prior = prior_cls(prior_dict)
        kwargs.pop('prior') if kwargs.get('prior', False) else None
        p_data = p_data_cls(prior=prior, N_prior=N_prior, **kwargs)
        
    path = path_class(p_simple, p_data)
    
    return path


def load_trained_model(job_id: str, device=None, ema=False):
    """
    Loads trained vector field, classifier and config.

    If model was trained with fixed N, returns None for classifier. 

    Args:
     job_id (str)
        unique id identifying run in checkpoints folder
     device (torch.device)
        device trained model will be put onto
     ema (bool)
        load exponentially averaged version of the model

    Returns:
        vector_field (nn.Module), classifier (nn.Module | None), config (dict)
    """
    # find the right directory through job_id
    run_dir = find_run_dir(
        job_id=job_id,
        save_dir="../checkpoints/"
    )
    checkpoint_path = os.path.join(run_dir, 'training_checkpoint.pth')
    config_path     = os.path.join(run_dir, 'config.yaml')

    # create empty model from config
    config = read_config(config_path)

    vector_field = empty_model_from_config(config)
    classifier = None

    transdimensional = not config['training']['fixed_N'] 
    if transdimensional:
        classifier = empty_classifier_from_config(config)
        vector_field = TransdimensionalModel(classifier, vector_field)
    
    # Load 'normal' or EMA version into empty model  
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if ema:
        EMA_checkpoint_path = os.path.join(run_dir, "EMA_checkpoint.pth")
        state_dict = torch.load(EMA_checkpoint_path, weights_only=False)
        vector_field.load_state_dict(state_dict) 
    else:
        vector_field.load_state_dict(checkpoint["model_state_dict"])

    vector_field.eval()
    vector_field.to(device)
    return vector_field, classifier, config


def load_model_info(config: dict, path: ConditionalProbabilityPath, device=None):
    """
    Load some info from config file into dict for easy access. 
    """
    # parameters to infer and (max) number of components in training data
    inf_params = config['model']['init_params']['inf_params']
    N = path.p_data.model_params['ncomp']

    # dimension of parameter vector 
    vector_dim = N * len(inf_params)

    # was N fixed during training?
    transdimensional = not config['training']['fixed_N'] 

    # standard setting of burst parameters
    burstparams = path.p_data.model_params['burstparams']
    mean, std = torch.tensor(config['training']['sample_mean'], device=device), torch.tensor(config['training']['sample_std'], device=device)

    return {
        "inf_params":inf_params,
        "N": N,
        "vector_dim":vector_dim,
        "burstparams":burstparams,
        "mean":mean,
        "std":std,
        "transdimensional":transdimensional
    }