import numpy as np
import torch 
import torch.nn as nn

from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import tqdm

from src.flow_matching.probability_path import ConditionalProbabilityPath
from src.flow_matching.helpers import model_size_b

class Trainer(ABC):
    """
    Generic trainer base class.
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

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, save_checkpoint=True, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / self.MiB:.3f} MiB')
        
        # Initialize
        self.path.to(device)
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        losses = np.zeros(num_epochs)

        # Train loop
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            opt.zero_grad()

            loss = self.get_train_loss(device, **kwargs)
            losses[epoch] = loss
            loss.backward()

            opt.step()

            # save checkpoint every 100 epochs
            if save_checkpoint and (epoch+1) % 100 == 0:
                self.save_checkpoint(epoch, opt, losses)

            progress_bar.set_description(f'Epoch {epoch}, loss: {loss.item():.3f}')

        # Finish
        self.model.eval()
        return losses
    
    def save_checkpoint(self, epoch, optimizer, losses, path="../checkpoints/"):
        """
        Saves training checkpoint.
        """
        timestamp = datetime.now().strftime('%d-%m_%H:%M:%S')
        save_dict = {
            "epoch": epoch,
            "model_state_dict":self.model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "losses":losses,
            "timestamp":timestamp
        }
        
        filename = f"training_checkpoint"
        torch.save(save_dict, path + filename + '.pth')

class ConditionalFlowMatchingTrainer(Trainer):

    """
    Trainer for unguided flow matching.
    """
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # samples
        z_batch = self.path.p_data.sample(batch_size) # z ~ p_data
        t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1)
        x_batch = self.path.sample_conditional_path(z_batch, t_batch) # x ~ p(x|z)

        # we take a monte carlo estimate of the loss:
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

        return average
    
class GuidedConditionalFlowMatchingTrainer(Trainer):

    """
    Trainer for guided flow matching model.
    """
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, device: torch.device, batch_size: int) -> torch.Tensor:
        # Only change here is label y should be an input to the learned vecotr field
        # samples
        z_batch, y_batch = self.path.p_data.sample(batch_size) # z, y ~ p_data
        t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1)
        x_batch = self.path.sample_conditional_path(z_batch, t_batch) # x ~ p(x|z)

        # put data on the doomsday device
        z_batch, y_batch = z_batch.to(device), y_batch.to(device)
        t_batch = t_batch.to(device)
        x_batch = x_batch.to(device)

        # we take a monte carlo estimate of the loss
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

        return average