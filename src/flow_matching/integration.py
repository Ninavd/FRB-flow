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
