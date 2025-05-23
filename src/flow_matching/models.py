from typing import List
import torch 
from src.flow_matching.helpers import build_mlp

class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        """
        Args:
        - dim: dimension of the parameters
        - hiddens: list of hidden layer sizes
        """
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)

class MLPGuidedVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x | y)
    """
    def __init__(self, dim: int, hiddens: List[int], y_dim: int=1000):
        """
        Args:
        - dim: number of parameters
        - hiddens: list of hidden layer sizes
        - y_dim: length of light curve
        """
        super().__init__()
        self.dim = dim
        self.y_dim = y_dim # length of lightcurve

        # input size is parameter dimension + time value + length of light curve (condition dimension)
        input_size = dim + 1 + self.y_dim 
        
        # output size is parameter dimension
        output_size = dim 

        self.net = build_mlp([input_size] + hiddens + [output_size])

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, dim) 
        - t: (bs, 1)
        - y: (bs, y_dim) [the condition tensor]
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t, y], dim=-1)
        return self.net(xt)
