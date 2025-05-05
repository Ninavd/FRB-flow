from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import torch 
import torch.nn as nn
from torch.func import vmap, jacrev
from tqdm import tqdm
from src.flow_matching.probability_path import ConditionalProbabilityPath

##################
# TRAINING
#################

def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)

class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)

class MLPGuidedVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int], y_dim=1000):
        super().__init__()
        self.dim = dim
        self.y_dim = y_dim # length of lightcurve

        # input size is parameter dimension + time value + length of light curve (condition dimension)
        input_size = dim + 1 + self.y_dim 
        
        # output size is parameter dimension
        output_size = dim 

        self.net = build_mlp([input_size] + hiddens + [output_size])

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, dim) 
        - t: (bs, 1)
        - y: (bs, y_dim) [the condition tensor]
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t, y], dim=-1)
        return self.net(xt)

MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')

        # Finish
        self.model.eval()

class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.i = 0

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # samples
        z_batch = self.path.p_data.sample(batch_size) # z ~ p_data
        t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1)
        x_batch = self.path.sample_conditional_path(z_batch, t_batch) # x ~ p(x|z)

        # monte carlo estimate of loss
        # 1/batch size * sum ((trained vector field) - (target vector field))**2
        
        differences = (
            self.model(x_batch, t_batch)
            - self.path.conditional_vector_field(x_batch, z_batch, t_batch)
            ) # shape batch_size, ndim

        losses = torch.sum(
            differences ** 2,
            dim = 1 # sum column wise
        )
        total_loss = torch.sum(losses)

        average =  total_loss / batch_size

        self.i += 1
        if self.i == 1:
          print(average.shape, losses.shape, batch_size, z_batch[0])

        return average
    
class GuidedConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.i = 0

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Only change here is label y should be an input to the learned vecotr field
        # samples
        z_batch, y_batch = self.path.p_data.sample(batch_size) # z, y ~ p_data
        t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1)
        x_batch = self.path.sample_conditional_path(z_batch, t_batch) # x ~ p(x|z)

        # monte carlo estimate of loss
        # 1/batch size * sum ((trained vector field) - (target vector field))**2
        
        differences = (
            self.model(x_batch, t_batch, y_batch)
            - self.path.conditional_vector_field(x_batch, z_batch, t_batch)
            ) # shape batch_size, ndim

        losses = torch.sum(
            differences ** 2,
            dim = 1 # sum column wise
        )
        total_loss = torch.sum(losses)

        average =  total_loss / batch_size

        self.i += 1
        if self.i == 1:
          print(average.shape, losses.shape, batch_size, z_batch[0])

        return average