import torch
import torch.nn.functional as F
from torch import nn

from src.flow_matching.helpers import build_mlp


class MLPVectorField(nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^phi(x)
    """
    def __init__(self, dim: int, hiddens: list[int]):
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
    def __init__(self, dim: int, hiddens: list[int], time_dim: int = 1000, 
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

        elif combine == "add":
            raise NotImplementedError()
        
        else:
            raise ValueError("INCORRECT COMBINATION METHOD. MUST BE [concat, add]")            

    def forward(self, x: torch.Tensor, tau:torch.Tensor, y: torch.Tensor, *args, **kwargs):
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
    Encodes input before classifying.
    Outputs logits (not normalized).
    """
    def __init__(self, encoder:nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder if encoder else lambda x : x
        self.config = self.make_config(**kwargs)
    
    def forward(self, y):
        y = self.net(self.encoder(y))
        return y

    def sample(self, y):
        p_N = self(torch.softmax(y, dim=1)) # (bs, N_max)
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

class TransdimensionalModel(nn.Module):
    """
    wrapper such that optimizer includes both models.
    """
    def __init__(self, classifier: nn.Module, vector_field_model: nn.Module):
        super().__init__()
        self.component_classifier = classifier
        self.vector_field_model = vector_field_model
    
    def forward(self, x, t, y, N):
        return self.vector_field_model(x, t, y, N)

if __name__=="__main__":
    import numpy as np
    # encoder = FRBLightCurveCNN()
    t = torch.linspace(0, 1, 1)
    t_batch = t.repeat(10, 1)
    print(t_batch.shape)
    result = fourier_embedding(t_batch, 128)
    print(result, result.shape)