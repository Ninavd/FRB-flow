import numpy as np
import torch 
import torch.nn as nn
from src.flow_matching.helpers import build_mlp
from src.flow_matching.distributions import Posterior, Prior
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.models import fourier_embedding

class TransformerGuidedField(nn.Module):
    def __init__(self, dim: int, inf_params: list[str], time_dim: int = 1000, 
                 time_seq_encoder: nn.Module | None = None, tau_encoder=None, theta_encoder=None):
        """
        Args:
        - dim: number of parameters
        - time_dim: (latent) length of light curve
        """
        super().__init__()
        self.dim   = dim
        self.inf_params = inf_params

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

        token_dim = self.theta_dim + self.time_dim + self.tau_dim

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=1, 
            dim_feedforward=256, 
            dropout=0.1, 
            activation=nn.GELU(), 
            batch_first=True, 
            norm_first=True, # TODO: set to False? 
            bias=True, 
            device=None
            )        
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=3
            )

        # tokens pass through down projection independently
        burst_params = 1  # TODO: edit when adding more params
        self.down_proj = nn.Linear(token_dim, burst_params) 

    def forward(self, x: torch.Tensor, tau:torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, dim) 
        -tau: (bs, 1)
        - y: (bs, time_dim) 
        Returns:
        - u_t^phi(x): (bs, dim)
        """
        # do encodings
        y   = self.time_seq_encoder(y)
        tau = self.tau_encoder(tau, self.tau_dim) 

        # split up x into N chunks of burstparams
        bs, dim = x.shape
        N_bursts = x.shape[-1] // len(self.inf_params)
        x = x.reshape(bs, N_bursts, dim // N_bursts) # (bs, dim) --> (bs, N, dim/N) TODO: only works when there's one param type 

        # put chunks of x through encoder (independently) 
        x   = self.theta_encoder(x) # (bs, N_bursts, N_params) (think this should be fine, MLP takes final dimension as input (?))
        
        # prepare y and tau to become part of token vectors (bs, N_tokens, token_dim) N_tokens = N_bursts
        y = y.unsqueeze(1)            # (bs, ≥1)    -> (bs, 1, ≥1)
        y = y.repeat(1, N_bursts, 1)  # (bs, 1, ≥1) -> (bs, N, ≥1)

        tau = tau.unsqueeze(1)                # (bs, ≥1)    -> (bs, 1, ≥1)
        tau = tau.repeat(1, N_bursts, 1)      # (bs, 1, ≥1) -> (bs, N, ≥1)

        # encoder input (bs, N_tokens, token_dim)
        tokens = torch.cat([y, x, tau], dim=-1) 
        _, _, token_dim = tokens.shape

        # add positional encoding
        positional_encoding = sinusoidal_PE(N_bursts, token_dim, tokens.device).unsqueeze(0).repeat(bs, 1, 1)
        tokens += positional_encoding

        # go through encoder and project down
        encoder_output = self.encoder(tokens)   # (bs, N, latent_dim)
        down_projected = self.down_proj(encoder_output) # (bs, N, N_inf_params)
        final_output    = down_projected.view(bs, dim) # (bs, N * N_inf_params)

        return final_output 
    
    def get_config(self):
        """
        Return dict with model setting for config file.
        """
        config = {
            "name":self._get_name(),
            "init_params":
            {
                "dim":self.dim,
                "time_dim":self.time_dim,
            }
        }
        return config

def sinusoidal_PE(N: int, d_model: int, device=None) -> torch.Tensor:
    """
    Sinusoidal positional encoding.

    Args:
        N (int): number of tokens.
        d_model (int): token dimension.
        device (torch.device): [optional] location of returned tensor.
    """
    position = torch.arange(N, dtype=torch.float).unsqueeze(1)  # (N, 1)

    # original formula: 1 / 10000^(2i / d_model) (edited for stability)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2, )

    # even indices get sin, uneven get cos
    pos_encoding = torch.zeros(N, d_model, device=device)
    pos_encoding[:, ::2]  = torch.sin(position * div_term) 
    pos_encoding[:, 1::2] = torch.cos(position * div_term) 

    return pos_encoding  

if __name__=="__main__":
    path = GuidedLinearProbabilityPath(
        p_simple=Prior(),
        p_data=Posterior()
    )
    dim = 2
    theta_encoder = lambda theta_dim : build_mlp([1, 8, 32, 64])
    time_encoder = build_mlp([1000, 512, 256, 128])
    tau_encoder = fourier_embedding

    batch_size=32
    z_batch, y_batch = path.p_data.sample(batch_size) # z, y ~ p_data
    x0_batch = path.p_simple.sample(batch_size) # x_0 ~ p_simple
    t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1) 
    x_batch = path.sample_conditional_path(x0_batch, z_batch, t_batch)

    transformer = TransformerGuidedField(dim=2, time_dim=128, time_seq_encoder=time_encoder, tau_encoder=tau_encoder, theta_encoder=theta_encoder)
    transformer(x_batch.cpu(), t_batch.cpu(), y_batch.cpu())