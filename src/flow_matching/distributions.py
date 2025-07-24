from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import torch
import torch.distributions as D

from src.flow_matching.simulator import Model
from src.flow_matching.helpers import choose_device

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

class UniformPrior(Sampleable):

    def __init__(self, x_min, x_max, log: bool, enforce_order: bool, dim: int):
        self.x_min = x_min if not log else np.log10(x_min)
        self.x_max = x_max if not log else np.log10(x_max)
        self.log = log # indicates if sample represents a power exponent
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
        
        samples = (self.x_max - self.x_min) * torch.rand(size=shape) + self.x_min 
        
        if self.enforce_order:
            sorted_samples, _ = torch.sort(samples, dim=1)
            return sorted_samples
        
        return samples
    
    def get_config(self):
        config = {
            "name":self.__class__.__name__, 
            "init_params":
            {
                "x_min"        :self.x_min,
                "x_max"        :self.x_max,
                "log"          :self.log,
                "enforce_order":self.enforce_order,
                "dim"          :self.dim
            }           
        }
        return config
    
class CompositePrior(Sampleable):
    def __init__(self, priors: dict[UniformPrior]):
        self.priors = priors
        self.dim = sum([priors[key].dim for key in priors])

    def sample(self, num_samples):
        """
        Sample composite prior. 
        Samples from each prior and concatenates result in a Tensor.

        Returns: 
            torch.Tensor: (bs, dim)
        """
        samples = torch.zeros((num_samples, self.dim))
        cursor = 0
        for key in self.priors:
            prior = self.priors[key]
            partial_sample = prior.sample(num_samples)
            samples[:, cursor : cursor + prior.dim] = partial_sample
            cursor = cursor + prior.dim
        return samples
    
    def sample_one_prior(self, name, num_samples):
        """
        Return samples from one of the priors. 
        """
        return self.priors[name].sample(num_samples)
    
    def samples_as_dict(self, samples):
        """
        Returns samples as a dictionary.

            dict[str, torch.Tensor(bs, prior_dim)]
        """
        # samples (bs, dim) --> split into 4 (bs, dim // len(priors))
        split_samples = torch.split(samples, self.dim // len(self.priors), dim =1)
        sample_dict = {key : samples for key, samples in zip(self.priors, split_samples)}
        return sample_dict
    
    def get_config(self):
        priors_config = []
        for prior in self.priors.values():
            priors_config.append(prior.get_config())

        config = {
            "name":self.__class__.__name__,
            "init_params": priors_config
        }

        return config

class Posterior(Sampleable):

    """
    Samples z, y ~ p(z)p(y|z), where z=model params, y=simulated data.
    """

    def __init__(self, model_params, inf_params, prior: CompositePrior):
        super().__init__()
        
        self.model_params = model_params # fixed burst parameters
        self.inf_params = inf_params
        self.prior = prior
        self.device = choose_device()
    
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
        burstparams = self.edit_burstparams(burstparams, self.prior.samples_as_dict(prior_samples))

        simulations = self.light_curve_sample(burstparams=burstparams)

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
        model = model.get_flux()
        model = model.to(self.device)
        x_counts = torch.poisson(model)

        return torch.Tensor(x_counts)
    
    def edit_burstparams(self, burstparams, prior_sample):
        """
        Edit burstsparams dict with prior samples
        """
        for key in prior_sample:
            burstparams[key] = prior_sample[key]
        return burstparams
    
    def get_config(self):
        # remove time from model params
        model_params = self.model_params.copy()
        model_params.pop('time')

        # make burstparams save-able
        model_params['burstparams'] = {
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in model_params['burstparams'].items()
        }

        config = {
            "name":self.__class__.__name__,
            "init_params":
            {
                "model_params": model_params,
                "inf_params": self.inf_params,
                "prior": self.prior.__class__.__name__
            }
        }
        return config

# Samplelable prior for t0_1 and t0_2
class PeaktimePrior(Sampleable):

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 2)
        """
        t0_samples, _ = torch.sort(torch.rand((num_samples, 2)), dim=1)
        return t0_samples
    
    def get_config(self):
        return self.__class__.__name__

# Samplelable posterior needs to sample prior and generate simulated data. 
class PeaktimePosterior(Sampleable):

    """
    Samples z, y ~ p(z)p(y|z), where z=model params, y=simulated data.
    """

    def __init__(self):
        super().__init__()

        # fixed burst parameters
        self.N = 2
        self.time = torch.linspace(0, 1.0, 1000)
        self.ybkg = 5.0
        
        # fixed component parameters
        amp  = 100.0
        rise = 0.03
        skew = 5
        
        # parameters of each burst component
        self.burstparams = {
            't0'   : [None, None], # to be sampled from prior
            'amp'  : torch.Tensor([amp, amp]),
            'rise' : torch.Tensor([rise, rise]),
            'skew' : torch.Tensor([skew, skew])
        }
    
    def sample(self, num_samples: int, prior) -> Tuple[torch.Tensor]:
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
        
        burstparams['t0'] = prior_samples
        simulations = self.light_curve_sample(burstparams=burstparams)

        return prior_samples, simulations
    
    def light_curve_sample(self, **kwargs) -> torch.Tensor:
        """
        Simulates lightcurve for given parameters.
        """
        # initialize burst model
        model = Model(time=self.time, ncomp=self.N, ybkg=self.ybkg, **kwargs)
        
        # simulate noisy light curve
        model = model.get_flux()
        model = model.to('cuda') if torch.cuda.is_available() else model
        simulated_counts = torch.poisson(model)

        return simulated_counts
    
    def get_config(self):
        return self.__class__.__name__

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