import torch.nn as nn 
from typing import List, Type

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