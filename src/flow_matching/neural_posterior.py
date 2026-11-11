import torch

from src.flow_matching.helpers import choose_device
from src.flow_matching.integration import EulerODESolver


class NeuralPosterior:
    """
    class to generate samples from the neural trans-dimensional FM posterior distribution.

    N, theta ~ p_xi(N | y) p_phi(theta | y, N)
    

    If the model had fixed N, it will only sample theta;
    theta ~ p(theta | y, N=fixed_N).
    """
    def __init__(self, vector_field, classifier, model_info, path, device=None, fixed_N=None):
        self.vector_field = vector_field
        self.classifier = classifier
        self.model_info = model_info
        self.path = path
        self.fixed_N = fixed_N

        if self.fixed_N:
            self.vector_field.N_max = fixed_N
            self.vector_field.vector_field_model.N_max = fixed_N
        self.device = device if device else choose_device()

    def sample(self, num_samples, x, samples_per_batch=None, time_steps=200, solver=EulerODESolver, **kwargs):
        """
        Generate parameter samples N, theta ~ p_xi(N | x) p_phi(theta | x, N)
        
        Args:
            num_samples (int): 
                number of samples to draw
            x (1000,): 
                condition (*unscaled* light curve).
            samples_per_batch (int): 
                (optional) number of samples per batch (use when running out of memory). default is `num_samples`
            time_steps (int): 
                (optional) number of timesteps used in numerical integration of flow matching field. default is 200
            solver (obj): 
                type of ODE solver to use (Euler, RK4 etc.) 
        returns:
            N_samples, theta_samples OR theta_samples if N is fixed (lightcurve assumed to have N components). 
        """
        
        condition = x
        try:
            num_samples = num_samples[0]
        except TypeError:
            pass

        samples_per_batch = samples_per_batch if samples_per_batch else num_samples
        batches = num_samples // samples_per_batch

        # allocate space for samples
        vector_dim = self.model_info["vector_dim"] if not self.fixed_N else 4 * self.fixed_N

        theta_samples = torch.zeros((batches * samples_per_batch, vector_dim), device=self.device)
        Ns = torch.zeros((batches * samples_per_batch, 1), device=self.device)

        # condition all prior samples on the same burst profile
        condition = torch.tensor(condition / 150, device=self.device, dtype=torch.float)
        simulations = condition.repeat(samples_per_batch, 1)
   
        self.p_N = self.component_posterior(condition.unsqueeze(0))
        solver = solver(self.vector_field)

        ts = torch.linspace(0, 1, time_steps).to(self.device)

        # integrate in batches
        for i in range(batches):
            
            N_samples, x1 = self.integrate(
                solver, samples_per_batch, simulations, ts, time_steps
                )

            start = i * samples_per_batch
            stop = start + samples_per_batch

            Ns[start:stop, :] = N_samples
            theta_samples[start:stop, :] = x1
        
        if self.fixed_N:
            return theta_samples.cpu()
        return Ns, theta_samples.cpu()

    def integrate(self, solver, num_points, simulations, ts, time_steps):
        """
        Integrate FM vector field. 

        solver (obj): 
            type of ODE solver to use (Euler, RK4 etc.)
        num_points (int):
            number of points to integrate.
        simulations torch.Tensor (num_points, 1000):
            FM condition to pass to network.
        ts torch.Tensor (time_steps,):
            array of time points of trajectory between 0 and 1.
        time_steps (int): 
            number of timesteps used in numerical integration of flow matching field.

        Returns:
        N_sampels, theta_samples
        """
        N_samples = torch.multinomial(
                self.p_N.repeat(num_points, 1), 
                num_samples=1
                ) + 1

        x0 = self.path.p_simple.sample(num_points, Ns=N_samples).to(self.device)

        if self.fixed_N:
            x0 = x0[:, :self.fixed_N * 4]

        mean = self.model_info["mean"]
        std = self.model_info["std"] 
        
        if self.fixed_N:
            new_dim = self.fixed_N * 4
            mean = mean.view(4, self.model_info["N"])[:, :self.fixed_N].reshape(new_dim)  
            std = std.view(4, self.model_info["N"])[:, :self.fixed_N].reshape(new_dim)  

        # solve ODE starting from x0
        x1 = solver.solve(
                x0, 
                ts.view(1, time_steps, 1).expand(num_points, time_steps, 1), 
                y=simulations, 
                N=N_samples
                ) * std + mean

        return N_samples, x1

    def component_posterior(self, condition):
        """
        Returns probability vector p(N | condition) of shape (N,).
        if N is fixed, returns one-hot vector (p_i = 1 if i=fixed_N, else p_i=0)
        """
        if self.model_info["transdimensional"] and not self.fixed_N:
            logits = self.classifier(condition)
            p_N = torch.softmax(logits, dim=1).flatten()
        else:
            N = self.fixed_N if self.fixed_N else self.model_info["N"]
            p_N = torch.zeros(N, device=self.device)
            p_N[N-1] = 1    
        return p_N
