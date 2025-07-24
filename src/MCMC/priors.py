import numpy as np

class UniformPrior():
    """
    Class for uniform prior.
    """
    def __init__(self, x_min, x_max, log, enforce_order, dim):
        self.x_min = x_min if not log else np.log10(x_min)
        self.x_max = x_max if not log else np.log10(x_max)
        self.log = log # indicates if sample represents a power exponent
        self.enforce_order = enforce_order
        self.dim = dim

    def log_prob(self, x):
        """
        Return log probability of x.
        """
        # out of range 
        if (x < self.x_min).any() or (x > self.x_max).any():
            return -np.inf
        
        # check if first value should be smaller than second etc. etc.
        if self.enforce_order and not np.all(x[:-1] <= x[1:]):
            return -np.inf
        
        else:
            return 0
    
    def sample(self, num_samples):
        """
        Generate samples from the distribution.
        """
        shape = (num_samples, self.dim)
        
        samples = np.random.uniform(self.x_min, self.x_max, size=shape) 
 
        if self.enforce_order:
            return np.sort(samples, axis=1)
        
        return samples