from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import torch
import torch.distributions as D

from src.simulator import Model, BurstSimulator

class Sampleable(ABC):
    """
    Base class for distribution which can be sampled from
    """ 
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass

# Samplelable prior for t0_1 and t0_2
class Prior(Sampleable):

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 2)
        """
        t0_samples, _ = torch.sort(torch.rand((num_samples, 2)), dim=1)
        return t0_samples

# Samplelable posterior needs to sample prior and generate simulated data. 
class Posterior(Sampleable):

    """
    Samples z, y ~ p(z)p(y|z), where z=model params, y=simulated data.
    """

    def __init__(self):
        super().__init__()

        # fixed burst parameters
        self.N = 2
        self.time = np.linspace(0, 1.0, 1000)
        self.ybkg = 5.0
        
        # fixed component parameters
        amp  = 25.0
        rise = 0.03
        skew = 5
        
        # parameters of each burst component
        self.burstparams = {
            't0'   : [None, None], # to be sampled from prior
            'amp'  : [amp, amp],
            'rise' : [rise, rise],
            'skew' : [skew, skew]
        }
    
    def sample(self, num_samples: int, prior=Prior()) -> Tuple[torch.Tensor]:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 2)
            torch.Tensor: shape (num_samples, 100)
        """
        # t0 samples 
        prior_samples = prior.sample((num_samples))

        burstparams = self.burstparams
        simulation_time_resolution = 1000
        simulations = torch.zeros((num_samples, simulation_time_resolution))

        # TODO: find a cleaner way to do this
        # sample simulated data from prior samples
        for idx in range(num_samples):
            
            # samples new peaktimes
            prior_sample = prior_samples[idx]
            burstparams['t0'] = list(prior_sample.cpu().numpy())

            # simulate with new params and save to array
            simulated_lightcurve = self.light_curve_sample(burstparams=burstparams)
            simulations[idx, :] = simulated_lightcurve

        return prior_samples, simulations
    
    def light_curve_sample(self, **kwargs) -> torch.Tensor:
        """
        Simulates lightcurve for given parameters.
        """
        # initialize burst model
        model = Model(time=self.time, ncomp=self.N, ybkg=self.ybkg, **kwargs)
        
        # simulate noisy light curve
        simulator = BurstSimulator(model)
        x_counts = simulator.simulate_burst()
        x_counts = torch.Tensor(x_counts)

        return torch.Tensor(x_counts)

class Gaussian(torch.nn.Module, Sampleable):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)
    
class CheckerboardSampleable(Sampleable):
    """
    Checkboard-esque distribution
    """
    def __init__(self, device: torch.device, grid_size: int = 3, scale=5.0):
        """
        Args:
            noise: standard deviation of Gaussian noise added to the data
        """
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 3)
        """
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0,2).to(self.device)
        while samples.shape[0] < num_samples:
            # Sample num_samples
            new_samples = (torch.rand(num_samples,2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:,0] + self.scale) / grid_length) % 2 == 0 # (bs,)
            y_mask = torch.floor((new_samples[:,1] + self.scale) / grid_length) % 2 == 0 # (bs,)
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)
        return samples[:num_samples]