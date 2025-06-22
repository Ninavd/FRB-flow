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

class UniformPrior(Sampleable):

    def __init__(self, x_min, x_max, log: bool, enforce_order: bool, dim: int):
        self.x_min = x_min
        self.x_max = x_max
        self.log = log
        self.enforce_order = enforce_order
        self.dim = dim

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, dim)
        """
        shape = (num_samples, self.dim)
        
        if not self.log:
            samples = (self.x_max - self.x_min) * torch.rand(size=shape) + self.x_min 
        else:
            samples = torch.exp((np.log(self.x_max) - np.log(self.x_min)) * torch.rand(size=shape) + np.log(self.x_min))
        
        if self.enforce_order:
            sorted_samples, _ = torch.sort(samples, dim=1)
            return sorted_samples
        
        return samples

class CompositePrior(Sampleable):
    def __init__(self, priors: list[UniformPrior]):
        self.priors = priors
        self.dim = sum([prior.dim for prior in priors])

    def sample(self, num_samples):
        samples = torch.zeros((num_samples, self.dim))
        cursor = 0
        for prior in self.priors:
            partial_sample = prior.sample(num_samples)
            samples[:, cursor : cursor + prior.dim] = partial_sample
            cursor = cursor + prior.dim
        return samples

class NewPosterior(Sampleable):

    """
    Samples z, y ~ p(z)p(y|z), where z=model params, y=simulated data.
    """

    def __init__(self, model_params, inf_params, prior: Sampleable):
        super().__init__()
        
        # fixed burst parameters
        self.model_params = model_params
        self.inf_params = inf_params
        self.prior = prior
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor]:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 2)
            torch.Tensor: shape (num_samples, 100)
        """
        # parameter samples 
        prior_samples = self.prior.sample((num_samples))

        burstparams = self.model_params['burstparams']
        simulation_time_resolution = 1000 # len(self.model_params['time'])
        simulations = torch.zeros((num_samples, simulation_time_resolution))

        # TODO: find a cleaner way to do this
        # sample simulated data from prior samples
        for idx in range(num_samples):
            
            # samples new peaktimes
            prior_sample = prior_samples[idx]
            burstparams = self.edit_burstparams(burstparams, prior_sample)

            # simulate with new params and save to array
            simulated_lightcurve = self.light_curve_sample(burstparams=burstparams)
            simulations[idx, :] = simulated_lightcurve

        return prior_samples, simulations
    
    def light_curve_sample(self, burstparams) -> torch.Tensor:
        """
        Simulates lightcurve for given parameters.
        """
        # initialize burst model
        model = Model(
            time=self.model_params['time'], 
            ncomp=self.model_params['ncomp'], 
            ybkg=self.model_params['ybkg'], 
            burstparams=burstparams
            )
        
        # simulate noisy light curve
        simulator = BurstSimulator(model)
        x_counts = simulator.simulate_burst()
        x_counts = torch.from_numpy(x_counts)

        return torch.Tensor(x_counts)
    
    def edit_burstparams(self, burstparams, prior_sample):
        """
        Edit burstsparams dict from flat prior sample
        """
        # NOTE: this code assumes N is fixed during training
        N = self.model_params['ncomp']

        for i, key in enumerate(self.inf_params):
            param_sample = prior_sample[i * N : i * N + N]
            burstparams[key] = list(param_sample.cpu().numpy())
        
        return burstparams


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