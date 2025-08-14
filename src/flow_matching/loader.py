from src.flow_matching.models import MLPGuidedVectorField, FRBLightCurveCNN, LightCurveThinner, fourier_embedding, LightCurveMLP, UNetEncoder, TransdimensionalModel, EncodedClassifier
from src.flow_matching.transformer import TransformerGuidedField
from src.flow_matching.distributions import UniformPrior, CompositePrior, Posterior, DiscreteUniform
from src.flow_matching.probability_path import GuidedLinearProbabilityPath

import yaml
from src.flow_matching.helpers import build_mlp
import torch
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
    "LightCurveThinner": LightCurveThinner,
    "FRBLightCurveCNN": FRBLightCurveCNN,
    "LightCurveMLP":LightCurveMLP,
    "UNetEncoder":UNetEncoder
}

TAU_ENCODERS = {
    True : fourier_embedding,
    False : None 
}

THETA_ENCODERS = {
    # True : lambda theta_dim : build_mlp([2] + [2 * 4, 2 * 8] + [theta_dim]),
    True : lambda theta_dim : build_mlp([4] + [4 * 8, 4 * 32] + [theta_dim]),
    False : None
}

PATHS = {
    "GuidedLinearProbabilityPath": GuidedLinearProbabilityPath,
}

DISTRIBUTIONS = {
    "Posterior":     Posterior,
    "UniformPrior":  UniformPrior,
    "CompositePrior":CompositePrior,
    "NewPosterior":  Posterior
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
            print(prior['init_params'])
            prior_dict[param] = prior_cls(**prior['init_params'])

        p_simple = p_simple_cls(prior_dict)

    else:
        p_simple = p_simple_cls(**kwargs)
    
    p_data_name   = config["path"]["p_data"]["name"]
    p_data_cls = DISTRIBUTIONS[p_data_name]

    kwargs = config["path"]["p_data"]["init_params"]
    kwargs.pop('prior') if kwargs.get('prior', False) else None
    kwargs['model_params']['time'] = torch.linspace(0, 1, 1000)
    
    N_prior = config['training'].get('fixed_N', None)
    if N_prior:
        N_prior = DiscreteUniform(kwargs['model_params']['ncomp'], kwargs['model_params']['ncomp'])
    else:
        N_prior = None
    p_data = p_data_cls(prior=p_simple, N_prior=N_prior, **kwargs)
    path = path_class(p_simple, p_data)
    
    return path