import numpy as np
import os
import torch 
import torch.nn as nn
import yaml 

from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import tqdm
from ema_pytorch import EMA

from src.flow_matching.probability_path import ConditionalProbabilityPath
from src.flow_matching.helpers import model_size_b, create_run_folder

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

    def train(self, num_epochs: int, device: torch.device,  lr: float = 1e-3, clip: float=None, use_ema: bool=True, save_checkpoint=True, batch_size: int = 256, job_id=None, **kwargs) -> torch.Tensor:
        # report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / self.MiB:.3f} MiB')
        
        # initialize
        self.path.to(device)
        self.model.to(device)

        if use_ema:
            ema = EMA(
                self.model,
                beta= 0.9999,
                update_after_step=100,
                update_every = 10
            )

        opt = self.get_optimizer(lr)
        lr_cutoff = int(0.9 * num_epochs)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, lr_cutoff, eta_min=1e-6)
        
        self.model.train()
        losses = np.zeros(num_epochs)

        # (optional) create run folder and save settings
        if save_checkpoint:
            self.save_path = create_run_folder("../checkpoints", job_id)
            self.save_config_file(num_epochs, lr, clip, batch_size, self.save_path)
        
        # train loop
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            opt.zero_grad()

            loss = self.get_train_loss(device, batch_size, **kwargs)
            losses[epoch] = loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip) if clip else None

            opt.step()
            lr_scheduler.step() if epoch < lr_cutoff else None

            ema.update() if use_ema else None 

            # save checkpoint every 100 epochs
            if save_checkpoint and (epoch+1) % 100 == 0:
                self.save_checkpoint(epoch, opt, losses)
                self.save_ema_checkpoint(ema) if use_ema else None

            progress_bar.set_description(f'Epoch {epoch}, loss: {loss.item():.2e}, lr: {lr_scheduler.get_last_lr()[0]:.2e}')

        # save final models
        if save_checkpoint:
            self.save_checkpoint(epoch, opt, losses)
            self.save_ema_checkpoint(ema) if use_ema else None

        # finish
        self.model.eval()
        return losses
    
    def save_checkpoint(self, epoch, optimizer, losses):
        """
        Saves training checkpoint.
        """
        timestamp = datetime.now().strftime('%d-%m_%H:%M:%S')
        save_dict = {
            "epoch"                : epoch,
            "model_state_dict"     : self.model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "losses"               : losses,
            "timestamp"            : timestamp
        }
        
        filename = f"training_checkpoint"
        torch.save(save_dict, os.path.join(self.save_path, filename + '.pth'))

    def save_ema_checkpoint(self, ema):
        filename = f"EMA_checkpoint"
        torch.save(ema.ema_model.state_dict(), os.path.join(self.save_path, filename + '.pth'))

    def save_config_file(self, num_epochs, lr, clip, batch_size, path):
        """
        Save yaml with training and model settings
        """
        try:
            t_encoder_config = self.model.time_seq_encoder.get_config()
        except AttributeError:
            t_encoder_config = False

        config = {
            "model"           : self.model.get_config(),
            "time_seq_encoder": t_encoder_config,
            "theta_encoder"   : self.model.encode_theta,
            "tau_encoder"     : self.model.encode_tau,
            "training":
            {
                "num_epochs"   : num_epochs,
                "learning_rate": lr,
                "batch_size"   : batch_size,
                "gradient_clip": clip,
                "optimizer"    : "adam"
            },
            "path": {
                "name"    :self.path.get_config(),
                "p_simple":self.path.p_simple.get_config(),
                "p_data"  : self.path.p_data.get_config()
            }

        }

        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

class ConditionalFlowMatchingTrainer(Trainer):

    """
    Trainer for unguided flow matching.
    """
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, device, batch_size: int) -> torch.Tensor:
        # samples
        z_batch = self.path.p_data.sample(batch_size) # z ~ p_data
        t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1)

        # put data on the doomsday device
        z_batch = z_batch.to(device)
        t_batch = t_batch.to(device)

        x0_batch = self.path.p_simple.sample(batch_size)
        x0_batch = x0_batch.to(device)
        x_batch = self.path.sample_conditional_path(x0_batch, z_batch, t_batch) # x ~ p(x|z)

        # put data on the doomsday device
        x_batch = x_batch.to(device)

        # we take a monte carlo estimate of the loss:
        # 1/batch size * sum ((trained vector field) - (target vector field))**2
        
        differences = (
            self.model(x_batch, t_batch)
            - self.path.conditional_vector_field(x0_batch, z_batch)
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
       
        # samples
        z_batch, y_batch = self.path.p_data.sample(batch_size) # z, y ~ p_data
        x0_batch = self.path.p_simple.sample(batch_size) # x_0 ~ p_simple
        # t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1) 

        # t ~ t^(1/1+a) (inverse sampling)
        u = torch.rand(batch_size, 1)
        alpha = -0.25
        power = (1 + alpha) / (2 + alpha)
        t_batch = torch.pow(u, power)
        
        # put data on the doomsday device
        z_batch, y_batch = z_batch.to(device), y_batch.to(device)
        t_batch  = t_batch.to(device)
        x0_batch = x0_batch.to(device)

        x_batch = self.path.sample_conditional_path(x0_batch, z_batch, t_batch) # x ~ p(x|z)
        
        # put data on the doomsday device
        x_batch = x_batch.to(device)

        # we take a monte carlo estimate of the loss
        # 1/batch size * sum ((trained vector field) - (target vector field))**2
        predicted_field = self.model(x_batch, t_batch, y_batch)
        target_field    = self.path.conditional_vector_field(x0_batch, z_batch)

        differences = predicted_field - target_field # shape batch_size, ndim
        MSE         = torch.sum(differences ** 2, dim = 1) # sum column wise
        
        h = 0
        field_size_penalty = h * torch.sum(predicted_field ** 2, dim=1)

        losses = MSE + field_size_penalty

        return torch.mean(losses)
