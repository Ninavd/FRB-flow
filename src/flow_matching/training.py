from abc import ABC, abstractmethod
import torch 
import torch.nn as nn
from tqdm import tqdm

from src.flow_matching.probability_path import ConditionalProbabilityPath
from src.flow_matching.helpers import model_size_b

class Trainer(ABC):
    """
    Generic trainer class.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.MiB = 1024 ** 2 # max model size

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / self.MiB:.3f} MiB')
        
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
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
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
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
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