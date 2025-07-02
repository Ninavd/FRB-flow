from datetime import datetime
import os
import torch.nn as nn 
from typing import List, Type
import torch 

def choose_device():
    # choose and state device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    return device

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

def build_mlp(dims: List[int], activation: Type[nn.Module] = nn.SiLU):
    """
    Build multilayer perceptron and return it.

    Args:
    - dims: dimension of each layer
    - activation: Activation function used.
    """
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(nn.Linear(dims[idx], dims[idx + 1]))
        
        # no activation on output layer
        if idx < len(dims) - 2:
            mlp.append(activation())
    
    return nn.Sequential(*mlp)

def load_model(model, path):
    """
    Loads model from checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    losses = checkpoint["losses"]
    # NOTE: set model to evaluation mode if you dont train  it further
    return model, losses

def find_run_dir(job_id:str, save_dir: str) -> str | None:
    """
    Returns path to folder where run was saved.
    """
    for dir_name in os.listdir(save_dir):
        if job_id in dir_name:
            return os.path.join(save_dir, dir_name)
    return None

def create_run_folder(path, job_id):
    """
    Creates folder in path to save training checkpoints and config file.
    """
    # check path exists
    if not os.path.isdir(path):
        raise Exception(f"{path} does not exist! The cwd is {os.getcwd()}")
    
    # generate foldername w timestamp
    timestamp = datetime.now().strftime("%d_%m_%H%M")
    folder_name = "run_" + timestamp
    folder_name = f"{job_id}_" + folder_name if job_id else folder_name
    full_path = path + '/' + folder_name

    # create run folder
    os.makedirs(full_path, exist_ok=True)

    # return relative path
    return full_path