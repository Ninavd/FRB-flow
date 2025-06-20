import numpy as np

class UniformPrior():
    """
    Class for uniform prior.
    """
    def __init__(self, x_min, x_max, log, enforce_order, dim):
        self.x_min = x_min
        self.x_max = x_max
        self.log = log
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
        
        # log uniform 
        if self.log:
            L = -np.log(x) - np.log(np.log(self.x_max) - np.log(self.x_min))
            return np.sum(L)
        
        else:
            return 0
    
    def sample(self, num_samples):
        """
        Generate samples from the distribution.
        """
        shape = (num_samples, self.dim)
        
        if not self.log:
            samples = np.random.uniform(self.x_min, self.x_max, size=shape) 
        elif self.log:
            samples = np.exp(np.random.uniform(np.log(self.x_min), np.log(self.x_max), size=shape))
        
        if self.enforce_order:
            return np.sort(samples, axis=1)
        
        return samples