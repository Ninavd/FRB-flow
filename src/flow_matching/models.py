from typing import List
import torch 
import torch.nn as nn
from src.flow_matching.helpers import build_mlp

class MLPVectorField(nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^phi(x)
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
        - u_t^phi(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)

class MLPGuidedVectorField(nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^phi(x | y)
    """
    def __init__(self, dim: int, hiddens: List[int], time_dim: int = 1000, 
                 time_seq_encoder: nn.Module | None = None, tau_encoder=None, theta_encoder=None, combine="concat"):
        """
        Args:
        - dim: number of parameters
        - hiddens: list of hidden layer sizes
        - time_dim: (latent) length of light curve

        embedding -> concat, add or GLU
        """
        super().__init__()
        self.dim   = dim

        # encoding dimensions
        self.time_dim  = time_dim # length of (encoded) lightcurve
        self.tau_dim   = time_dim // 2 if tau_encoder else 1
        self.theta_dim = time_dim // 2 if theta_encoder else dim 

        # encoders
        self.time_seq_encoder = time_seq_encoder if time_seq_encoder else lambda x : x
        self.tau_encoder      = tau_encoder      if tau_encoder      else lambda x, _ : x
        self.theta_encoder    = theta_encoder(self.theta_dim) if theta_encoder else lambda x : x 
        
        self.combine = combine 

        # output size is parameter dimension
        output_size = dim 
        self.hiddens = hiddens

        if combine == "concat":
            input_size = self.theta_dim + self.tau_dim + self.time_dim 
            self.net = build_mlp([input_size] + hiddens + [output_size]) 

        elif combine == "GLU":
            self.net = GLUInjectedMLP(input_dim=self.time_dim, cond_dim=self.theta_dim + self.tau_dim, hiddens=hiddens, output_dim=dim)

        elif combine == "add":
            raise NotImplementedError()
        
        else:
            raise ValueError("INCORRECT COMBINATION METHOD. MUST BE [concat, add, GLU]")            

    def forward(self, x: torch.Tensor, tau:torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, dim) 
        -tau: (bs, 1)
        - y: (bs, time_dim) [the condition tensor]
        Returns:
        - u_t^phi(x): (bs, dim)
        """
        # do encodings
        y   = self.time_seq_encoder(y)
        tau = self.tau_encoder(tau, self.tau_dim)
        x   = self.theta_encoder(x)
        
        # define the condition vector
        condition = torch.cat([x, tau], dim=-1)

        # put (combinations of) encodings through network
        if self.combine == "concat":
            x_tau = torch.cat([condition, y], dim=-1) 
            return self.net(x_tau)
        
        elif self.combine == "GLU":
            return self.net(y, condition)
        
        elif self.combine == "add":
            raise NotImplementedError()
    
    def get_config(self):
        """
        Return dict with model setting for config file.
        """
        config = {
            "name":self._get_name(),
            "init_params":
            {
                "dim":self.dim,
                "hiddens": self.hiddens,
                "time_dim":self.time_dim,
                "combine":self.combine
            }
        }
        return config

class GLUInjectLayer(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=None):
        super().__init__()

        hidden_dim = hidden_dim or input_dim
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_proj = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        x_proj = self.activation(self.x_proj(x))  
        gate = torch.sigmoid(self.gate_proj(cond))  
        return x_proj * gate  

class GLUInjectedMLP(nn.Module):
    def __init__(self, input_dim, cond_dim, hiddens: List[int], output_dim):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hiddens[0])

        self.layers = nn.ModuleList([
            GLUInjectLayer(hidden_dim, cond_dim, hiddens[i+1])
            for i, hidden_dim in enumerate(hiddens[:-1])
        ])

        self.output_proj = nn.Linear(hiddens[-1], output_dim)

    def forward(self, x, cond):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, cond)  # GLU + conditioning at each layer
        return self.output_proj(x)

class FRBLightCurveCNN(nn.Module):
    """
    Simple 1-D CNN encoder for ≈1000-sample FRB light-curves.
    Symmetric padding (“same”) + stride-2 down-sampling → length 1000 → 500 → 250 → 125 → 63.
    Returns a latent vector h(x)∈R^latent_dim.
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()

        self.channels  = [32, 64, 128, 128]   # channels per stage
        self.kernel  = 5
        self.stride  = 2
        self.padding = self.kernel // 2          # symmetric “same” padding
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        layers, c_in = [], in_channels
        for c_out in self.channels:
            layers += [
                nn.Conv1d(c_in, c_out,
                          kernel_size=self.kernel,
                          stride=self.stride,
                          padding=self.padding,     # symmetric
                          bias=False),
                nn.GELU(),
                nn.BatchNorm1d(c_out),
            ]
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool1d(1)   # (B,C_last,1)
        self.proj     = nn.Linear(self.channels[-1], latent_dim)

    def forward(self, x):                         # x: (B,C,≥1)
        x = x.unsqueeze(1) 
        y = self.backbone(x)                      # (B,C_last,L≈63)
        y = self.pool(y).squeeze(-1)              # (B,C_last)
        return self.proj(y)                       # (B,latent_dim)
    
    def get_config(self):
        """
        Return dict with model setting for config file.
        """
        config = {
            "name":self._get_name(),
            "init_params":
            {
                "in_channels": self.in_channels,
                "latent_dim": self.latent_dim,
            },
            "other settings":
            {
                "channels":self.channels,
                "kernel": self.kernel,
                "stride": self.stride,
                "padding": self.padding,
            }
        }
        return config

class LightCurveThinner(nn.Module):
    """
    Simple 'encoder' for time series.
    Thins it out by only keeping every N-th bin.
    """
    def __init__(self, latent_dim):
        assert 1000 % latent_dim == 0, 'Invalid latent dimension (time series length not divisible by latent_dim)'
        super().__init__()
        
        self.stride = 1000 // latent_dim

    def forward(self, x): # x: (B, ≥1)
        return x[:, ::self.stride]
    
    def get_config(self):
        """
        Return dict with model setting for config file.
        """
        config = {
            "name":self._get_name(),
            "init_params":
            {
                "latent_dim": 1000 // self.stride,
            }
        }
        return config
    
def fourier_embedding(tau, latent_dim):
    # log-spaced frequencies
    freqs = torch.logspace(0, 3, latent_dim // 2)
    freqs = freqs.to(tau.device)

    tau = tau.view(-1, 1)  # shape: (batch_size, 1)
    wt = tau * freqs
    return torch.cat([torch.sin(wt), torch.cos(wt)], dim=-1)


if __name__=="__main__":
    import numpy as np
    # encoder = FRBLightCurveCNN()
    t = torch.linspace(0, 1, 1)
    t_batch = t.repeat(10, 1)
    print(t_batch.shape)
    result = fourier_embedding(t_batch, 128)
    print(result, result.shape)