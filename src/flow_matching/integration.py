import torch 
from tqdm import tqdm

class EulerODESolver():

    """
    Solves ODE with Euler method.
    """

    def __init__(self, vector_field: torch.nn.Module):
        self.vector_field = vector_field
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)
            - dt: time, shape (bs, 1)
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        return xt + self.vector_field(xt,t, **kwargs) * h

    @torch.no_grad()
    def solve(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, dim)
            - ts: timesteps, shape (bs, nts, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, dim)
        """
        # number of time steps
        nts = ts.shape[1]

        # tqdm prints progress bar
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def solve_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, dim)
            - ts: timesteps, shape (bs, nts, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, dim)
        """
        # add initial state
        xs = [x.clone()]  

        nts = ts.shape[1] # number of timesteps

        # tqdm prints progress bar
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)

            xs.append(x.clone())

        return torch.stack(xs, dim=1)

class ODEBackwardSolver():

    """
    Solves two ODEs backwards (t: 1 -> 0) with Euler method.
    This allows us to evaluate probability values of
    a set of samples.
    (see appendix C of original FM paper)
    """

    def __init__(self, vector_field: torch.nn.Module):
        self.vector_field = vector_field

    def step(self, xt, ft, t, h, **kwargs):
        """
        Takes one simulation step to find f(1-t) and x(1-t)
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)
            - dt: time, shape (bs, 1)
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        # h is negative for backwards integration
        x_prev = xt + h * self.vector_field(xt,t, **kwargs)
        f_prev = ft - h[0] * self.divergence(self.vector_field(xt, t), xt)
        return x_prev, f_prev

    def solve(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, dim)
            - ts: timesteps, shape (bs, nts, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, dim)
        """
        assert ts[0, 0, 0] > ts[0,-1,0], "Starttime must be bigger than endtime such that stepsize is negative!"

        # number of time steps
        nts = ts.shape[1]

        # initial value of f(1)
        f = torch.zeros(x.shape[0], device=x.device)
   
        # tqdm prints progress bar
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x, f = self.step(x, f, t, h, **kwargs)

        return x, f
    
    def f_to_posterior_log_prob(self, f_0, x_0):
        """
        Convert f(0) found with `solve` to a posterior probability.
        Assumes p_init (aka p_simple) is a standard normal N(0,1)
        """
        log_p_0 = torch.distributions.MultivariateNormal(
            torch.zeros_like(x_0, device=x_0.device), 
            torch.eye(x_0.shape[-1], device=x_0.device)
            ).log_prob(x_0) 
        return log_p_0 - f_0

    def divergence(self, velocity_vector, x):
        """
        Approximate divergence of the velocity.
        Adapted from: https://github.com/dingo-gw/dingo/blob/main/dingo/core/posterior_models/cflow_base.py
        """
        div = 0.0
        with torch.enable_grad():
            velocity_vector.requires_grad_(True)
            x.requires_grad_(True)
            # trace of jacobian
            for i in range(velocity_vector.shape[-1]):
                div += torch.autograd.grad(
                    velocity_vector[..., i], x, torch.ones_like(velocity_vector[..., i]), retain_graph=True
                )[0][..., i : i + 1]
            return div.view(div.shape[0])