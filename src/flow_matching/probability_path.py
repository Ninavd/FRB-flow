from abc import ABC, abstractmethod
from typing import Tuple
import torch 
import torch.nn as nn
from src.flow_matching.distributions import Sampleable 

###########################
# Conditional prob path
###########################

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, dim)
        """
        num_samples = t.shape[0]

        # Sample conditioning variable z ~ p(z)
        z = self.sample_conditioning_variable(num_samples)[0] 

        # sample from initial distribution
        x0 = self.p_simple.sample(num_samples)

        # move to same device
        device = t.device
        x0, z = x0.to(device), z.to(device)

        # sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(x0, z, t) # (num_samples, dim)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, dim)
            - y: (num_samples, label_dim)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, x0, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        pass

    def get_config(self):
        return self.__class__.__name__
    
class LinearConditionalProbabilityPath(ConditionalProbabilityPath):

    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, ...)
        """
        z = self.p_data.sample(num_samples) 
        return z, torch.zeros_like(z) # dummy return value

    def sample_conditional_path(self, x0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples the random variable X_t = (1-t) X_0 + tz
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        sigma_min = 1e-4
        return (1 - (1 - sigma_min) * t) * x0 + z * t

    def conditional_vector_field(self, x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z) = (z - x0)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """
        sigma_min = 1e-4
        return z - (1 - sigma_min) * x0

class GuidedLinearProbabilityPath(LinearConditionalProbabilityPath):
    
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__(p_simple, p_data)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
            """
            Samples the conditioning variable and label z, y ~ p_data(x)
            Args:
                - num_samples: the number of samples
            Returns:
                - z: samples from p(z), (num_samples, ...)
                - y: labels (num_samples, label_dim)
            """
            return self.p_data.sample(num_samples)      