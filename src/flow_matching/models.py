from typing import List
import torch 
import torch.nn as nn
import torch.nn.functional as F
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

        # encoders boolean (for config)
        self.encode_time_seq = True if time_seq_encoder else False
        self.encode_tau      = True if tau_encoder else False 
        self.encode_theta    = True if theta_encoder else False 

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

class LightCurveMLP(nn.Module):
    """
    MLP encoder for time series.
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.net = build_mlp(layers)
    
    def forward(self, x):
        return self.net(x)
    
    def get_config(self):
        config = {
            "name":self._get_name(),
            "init_params":
            {
                "layers": self.layers,
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

class UNetEncoder(nn.Module):
    """
    Encoder part of U-Net - maintains fine details through skip connections
    """

    def __init__(self, seq_len=1000, latent_dim=64):
        super().__init__()

        self.config = self.make_config(seq_len=seq_len, latent_dim=latent_dim)

        # Encoder path - gentler downsampling
        self.enc1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.enc2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  # L/2
        self.enc3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)  # L/4
        self.enc4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # L/8

        # Process at multiple scales
        self.process1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.process2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.process3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        # Multi-scale feature aggregation
        self.scale_proj1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(32, 64, kernel_size=1)
        )
        self.scale_proj2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(64, 64, kernel_size=1)
        )
        self.scale_proj3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(128, 64, kernel_size=1)
        )
        self.scale_proj4 = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(256, 64, kernel_size=1)
        )

        # Final projection
        self.final = nn.Sequential(
            nn.Linear(64 * 16 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):  # x: (B, L)
        x = x.unsqueeze(1) # (B, 1, L)

        # Encode with feature preservation
        h1 = F.relu(self.enc1(x))  # (B, 32, L)
        h1_proc = self.process1(h1)

        h2 = F.relu(self.enc2(h1))  # (B, 64, L/2)
        h2_proc = self.process2(h2)

        h3 = F.relu(self.enc3(h2))  # (B, 128, L/4)
        h3_proc = self.process3(h3)

        h4 = F.relu(self.enc4(h3))  # (B, 256, L/8)

        # Multi-scale aggregation
        feat1 = self.scale_proj1(h1_proc).flatten(1)
        feat2 = self.scale_proj2(h2_proc).flatten(1)
        feat3 = self.scale_proj3(h3_proc).flatten(1)
        feat4 = self.scale_proj4(h4).flatten(1)

        # Concatenate multi-scale features
        features = torch.cat([feat1, feat2, feat3, feat4], dim=1)

        return self.final(features)

    def get_config(self):
        return self.config
    
    def make_config(self, **kwargs):
        config = {
            "name": self._get_name(),
            "init_params":
            {
                **kwargs
            }
        }
        return config

class GenericClassifier(nn.Module):
    def __init__(self, inputs, hiddens, outputs, activation=nn.SiLU):
        super().__init__()

        self.net = build_mlp([inputs] + hiddens + [outputs], activation)
        self.softmax = nn.Softmax(dim=1) if outputs > 1 else nn.Sigmoid()

    def forward(self, x): # x: (bs, dim)
        x = self.net(x)
        x = self.softmax(x)
        return x          # (bs, 1)

class EncodedClassifier(GenericClassifier):
    """
    Encodes input before classifying
    """
    def __init__(self, encoder:nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.config = self.make_config(**kwargs)
    
    def forward(self, y):
        y = self.net(self.encoder(y))
        return self.softmax(y)

    def sample(self, y):
        p_N = self(y) # (bs, N_max)
        return torch.multinomial(p_N, num_samples=1) + 1  # N: (bs, 1)

    def make_config(self, **kwargs):
        config = {
            "name": self._get_name(),
            "init_params":
            {
                **kwargs
            },
            "encoder":
            {
                **self.encoder.get_config()
            }
        }
        return config
    
    def get_config(self):
        return self.config
    
if __name__=="__main__":
    import numpy as np
    # encoder = FRBLightCurveCNN()
    t = torch.linspace(0, 1, 1)
    t_batch = t.repeat(10, 1)
    print(t_batch.shape)
    result = fourier_embedding(t_batch, 128)
    print(result, result.shape)