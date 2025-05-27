from typing import List
import torch 
import torch.nn as nn
from src.flow_matching.helpers import build_mlp

class MLPVectorField(nn.Module):
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

class MLPGuidedVectorField(nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x | y)
    """
    def __init__(self, dim: int, hiddens: List[int], y_dim: int = 1000, time_seq_encoder: nn.Module | None = None):
        """
        Args:
        - dim: number of parameters
        - hiddens: list of hidden layer sizes
        - y_dim: length of light curve
        """
        super().__init__()
        self.dim   = dim
        self.y_dim = y_dim # length of lightcurve
        self.time_seq_encoder = time_seq_encoder

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
        if self.time_seq_encoder is not None:
            y = self.time_seq_encoder(y)

        xt = torch.cat([x, t, y], dim=-1)
        return self.net(xt)

class FRBLightCurveCNN(nn.Module):
    """
    Simple 1-D CNN encoder for ≈1000-sample FRB light-curves.
    Symmetric padding (“same”) + stride-2 down-sampling → length 1000 → 500 → 250 → 125 → 63.
    Returns a latent vector h(x)∈R^latent_dim.
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()

        widths  = [32, 64, 128, 128]   # channels per stage
        kernel  = 5
        stride  = 2
        padding = kernel // 2          # symmetric “same” padding

        layers, c_in = [], in_channels
        for c_out in widths:
            layers += [
                nn.Conv1d(c_in, c_out,
                          kernel_size=kernel,
                          stride=stride,
                          padding=padding,     # symmetric
                          bias=False),
                nn.GELU(),
                nn.BatchNorm1d(c_out),
            ]
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool1d(1)   # (B,C_last,1)
        self.proj     = nn.Linear(widths[-1], latent_dim)

    def forward(self, x):                         # x: (B,C,≥1)
        x = x.unsqueeze(1) 
        y = self.backbone(x)                      # (B,C_last,L≈63)
        y = self.pool(y).squeeze(-1)              # (B,C_last)
        return self.proj(y)                       # (B,latent_dim)

if __name__=="__main__":
    import numpy as np
    encoder = FRBLightCurveCNN()
    x = torch.linspace(0, 1, 10000)
    x_batch = x.repeat(10, 1)
    print(x_batch.shape)
    result = encoder(x_batch)
    print(result, result.shape)