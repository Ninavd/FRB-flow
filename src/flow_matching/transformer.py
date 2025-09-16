import numpy as np
import torch
import torch.nn as nn
from src.flow_matching.helpers import build_mlp
from src.flow_matching.distributions import PeaktimePosterior, PeaktimePrior
from src.flow_matching.probability_path import GuidedLinearProbabilityPath
from src.flow_matching.models import fourier_embedding


class TransformerGuidedField(nn.Module):

    """Flow matching vector field $u_t^phi(x|y, N)$ on transformer encoder architecture. 
    
    Tokens are constructed by concatenating the encoded time series,
    parameter vector and flow matching time. The sequence length is fixed,
    but tokens are masked out in accordance with the sampled sequence length 
    (i.e. burst components) passed in the forward call.  
    """

    def __init__(
            self, 
            dim: int, 
            inf_params: list[str], 
            time_dim: int = 1000, 
            time_seq_encoder: nn.Module | None = None,
            tau_encoder: nn.Module | None = None, 
            theta_encoder: nn.Module | None = None,
            encoder_kwargs = dict(
                nhead=4, 
                dim_feedforward=1024, 
                dropout=0.0),
            n_layers = 6,
            add_pos_encoding=True
            ):
        """Initializes the vector field.
   
        Args:
            dim (int): The dimension of the vector field.
            inf_params (list of str): Names of burst component parameters, f.e. `[\'t0\', \'amp\']`. 
            time_dim (int): Latent length of the encoded light curve.
            time_seq_encoder (nn.Module or None): Trainable encoder compressing the time series.
            tau_encoder (nn.Module or None): Should inflate flow matching time to half of `time_dim`.
            theta_encoder (nn.Module or None): Inflates parameter vector theta.
            encoder_kwargs (dict): arguments passed to `torch.nn.TransformerEncoderLayer`.
            n_layers (int): numeber of sequential encoder layers.
            add_pos_encoding (bool): Add positional encoding to tokens.
        """
        super().__init__()

        # encoding dimensions
        self.tau_dim   = time_dim // 2 if tau_encoder else 1
        self.theta_dim = time_dim // 2 if theta_encoder else dim

        # encoders boolean (for config)
        self.encode_time_seq = True if time_seq_encoder else False
        self.encode_tau      = True if tau_encoder else False 
        self.encode_theta    = True if theta_encoder else False 

        # encoders
        self.time_seq_encoder = time_seq_encoder if time_seq_encoder else lambda x: x
        self.tau_encoder      = tau_encoder      if tau_encoder      else lambda x, _: x
        self.theta_encoder    = theta_encoder(self.theta_dim) if theta_encoder else lambda x: x 

        token_dim = self.theta_dim + time_dim + self.tau_dim

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=token_dim,
            activation=nn.GELU(), 
            batch_first=True, 
            norm_first=True,# TODO: set to False? 
            **encoder_kwargs, 
        )        
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # maximum number of burst components  
        self.N_max = dim // len(inf_params)

        # tokens pass through down projection independently
        burst_params = len(inf_params)  
        self.down_proj = nn.Linear(token_dim, burst_params) 

        self.add_pos_encoding = add_pos_encoding
        self.config = self.make_config(encoder_kwargs, n_layers=n_layers, add_pos_encoding=add_pos_encoding, dim=dim, inf_params=inf_params, time_dim=time_dim)

    def prepare_tokens(self, x_e: torch.Tensor, tau_e: torch.Tensor, y_e: torch.Tensor) -> torch.Tensor:
        """Expand conditions to correct shape and concatenate to create tokens.
        
        Args:
            x_e:   encoded input (parameters).
            tau_e: encoded flow matching time.
            y_e:   encoded condition (lightcurve).
        
        Returns:
            torch.Tensor: Tokens of shape (bs, N_tokens, token_dim)
        """
        y_e = y_e.unsqueeze(1)              # (bs, ≥1)    -> (bs, 1, ≥1)
        y_e = y_e.repeat(1, self.N_max, 1)  # (bs, 1, ≥1) -> (bs, N, ≥1)

        tau_e = tau_e.unsqueeze(1)              # (bs, ≥1)    -> (bs, 1, ≥1)
        tau_e = tau_e.repeat(1, self.N_max, 1)  # (bs, 1, ≥1) -> (bs, N, ≥1)

        tokens = torch.cat([y_e, x_e, tau_e], dim=-1) 
        return tokens
    
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Split up x into N chunks and pass chunks through encoder independently. 
        
        Each token represents a burst component.

        Args:
            x (torch.Tensor): Parameter vector of shape (bs, dim).

        Returns:
            torch.Tensor: Tensor of shape (bs, N, theta_dim)
        """
        # x = [t0_1 .... t0_n, amp_1, amp_2, ... amp_n, ...]
        # -> view as 2D matrix and transpose to get:
        # x = [[t0_1, amp_1, rise_1, skew_1], .... .[t0_n, amp_n, rise_n, skew_n]]
        bs, dim = x.shape
        x = x.view(bs, dim // self.N_max, self.N_max).transpose(1, 2) 

        # MLP takes final dimension as input
        x   = self.theta_encoder(x) # (bs, N_bursts, theta_dim)
        return x
        
    def forward(self, x: torch.Tensor, tau: torch.Tensor, y: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   (torch.Tensor): Parameter values. Tensor of shape (bs, dim).
            tau (torch.Tensor): Flow matching time. Tensor of shape (bs, 1).
            y   (torch.Tensor): Condition vector (raw light curve). Tensor of shape (bs, raw_time_dim).
            N   (torch.Tensor): Number of burst components in y. Tensor of shape (bs, 1).

        Returns:
            $u_t^phi(x|y, N)$ (torch.Tensor): Tensor of shape (bs, dim)
        """
        # encode conditions
        y   = self.time_seq_encoder(y)
        tau = self.tau_encoder(tau, self.tau_dim) 

        # reshape and encode input
        bs, dim = x.shape
        x = self.prepare_input(x) 

        tokens = self.prepare_tokens(x, tau, y)
        _, _, token_dim = tokens.shape

        # add positional encoding
        if self.add_pos_encoding:
            positional_encoding = sinusoidal_PE(self.N_max, token_dim, tokens.device).unsqueeze(0).repeat(bs, 1, 1)
            tokens += positional_encoding

        # create mask of shape (bs, N_max) that is False for first N tokens
        mask = torch.arange(self.N_max, device=N.device).expand(len(N), self.N_max) >= N 

        # go through encoder and project down
        encoder_output = self.encoder(tokens, src_key_padding_mask=mask)   # (bs, N, latent_dim)
        down_projected = self.down_proj(encoder_output)                    # (bs, N, N_inf_params)
        
        # want final vector to be to [t0_1, ..., t0_n, amp_1, ..., amp_n, ... etc]
        down_projected = down_projected.transpose(1, 2)

        final_output = down_projected.reshape(bs, dim) # (bs, N * N_inf_params)
        
        return final_output 
    
    def make_config(self, encoder_kwargs, **kwargs):
        """
        Create dictionary with model settings.
        """
        config = {
            "name": self._get_name(),
            "init_params":
            {
                **kwargs,
                'encoder_kwargs': encoder_kwargs
            }
        }
        return config

    def get_config(self):
        """
        Returns dictionary with model settings.
        """
        return self.config

def sinusoidal_PE(N: int, d_model: int, device=None) -> torch.Tensor:
    """Sinusoidal positional encoding.

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
        p_simple=PeaktimePrior(),
        p_data=PeaktimePosterior()
    )
    dim = 2
    theta_encoder = lambda theta_dim : build_mlp([1, 8, 32, 64])
    time_encoder = build_mlp([1000, 512, 256, 128])
    tau_encoder = fourier_embedding

    batch_size=32
    z_batch, y_batch = path.p_data.sample(batch_size) # z, y ~ p_data
    x0_batch = path.p_simple.sample(batch_size)       # x_0 ~ p_simple
    t_batch = torch.rand(batch_size, 1)               # t ~ U(0, 1) 
    x_batch = path.sample_conditional_path(x0_batch, z_batch, t_batch)

    transformer = TransformerGuidedField(dim=2, time_dim=128, time_seq_encoder=time_encoder, tau_encoder=tau_encoder, theta_encoder=theta_encoder)
    transformer(x_batch.cpu(), t_batch.cpu(), y_batch.cpu())