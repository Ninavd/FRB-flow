from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import torch 
import torch.nn as nn
from torch.func import vmap, jacrev
from tqdm import tqdm
from src.flow_matching.probability_path import ConditionalProbabilityPath, Sampleable

class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass

