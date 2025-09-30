import numpy as np
import os
import torch 
import torch.nn as nn
import wandb
import yaml 

from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import tqdm
from ema_pytorch import EMA

from src.flow_matching.probability_path import ConditionalProbabilityPath
from src.flow_matching.models import TransdimensionalModel
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

    def init_lr_scheduler(self, opt, lr, lr_cutoff, warm_up_iters):
        # linearly increase multiplier to 1.0
        lr_start = 1e-8
        start_factor = lr_start / lr  
        slope = (1.0 - start_factor) / warm_up_iters 
        lin_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch : (epoch+1) * slope) 
        cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, lr_cutoff - warm_up_iters, eta_min=1e-6)

        return torch.optim.lr_scheduler.SequentialLR(opt, [lin_lr_scheduler, cos_lr_scheduler], milestones=[warm_up_iters])

    def train(
        self, 
        num_epochs: int, 
        device: torch.device, 
        lr: float = 1e-3,
        clip: float=None,
        use_ema: bool=True, 
        save_checkpoint=True, 
        batch_size: int = 256, 
        job_id=None, 
        fixed_N: bool=False,
        **kwargs
        ) -> torch.Tensor:
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

        warm_up_iters = min(500, int(0.1 * num_epochs))
        lr_cutoff     = int(0.95 * num_epochs) 
        lr_scheduler  = self.init_lr_scheduler(opt, lr, lr_cutoff, warm_up_iters)
        
        self.model.train()
        losses = np.zeros(num_epochs)
        MSE_losses = np.zeros(num_epochs)
        CEL_losses = np.zeros(num_epochs)

        # (optional) create run folder and save settings
        if save_checkpoint:
            self.save_path = create_run_folder("../checkpoints", job_id)
            self.save_config_file(num_epochs, lr, clip, batch_size, self.save_path, fixed_N, **kwargs)
        
        # train loop
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            opt.zero_grad()
            
            if not fixed_N:
                MSE, CEL = self.get_train_loss(device, batch_size, **kwargs)
                MSE_losses[epoch] = MSE
                CEL_losses[epoch] = CEL
                loss = MSE + CEL
            else:
                loss = self.get_train_loss(device, batch_size, **kwargs)
                MSE, CEL = loss, 0

            losses[epoch] = loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip) if clip else None

            opt.step()
            lr_scheduler.step() if epoch < lr_cutoff else None

            ema.update() if use_ema else None 

            # save checkpoint every 100 epochs
            if save_checkpoint:
                wandb.log({'loss':loss, 'MSE_loss':MSE, 'CEL_loss':CEL, 'lr':lr_scheduler.get_last_lr()[0]})
            if save_checkpoint and (epoch+1) % 100 == 0:
                self.save_checkpoint(epoch, opt, losses, MSE_losses, CEL_losses)
                self.save_ema_checkpoint(ema) if use_ema else None

            progress_bar.set_description(f'Epoch {epoch}, loss: {loss.item():.2e}, lr: {lr_scheduler.get_last_lr()[0]:.2e}')

        # save final models
        if save_checkpoint:
            self.save_checkpoint(epoch, opt, losses, MSE_losses, CEL_losses)
            self.save_ema_checkpoint(ema) if use_ema else None

        # finish
        self.model.eval()
        return losses
    
    def save_checkpoint(self, epoch, optimizer, losses, MSE_losses, CEL_losses):
        """
        Saves training checkpoint.
        """
        timestamp = datetime.now().strftime('%d-%m_%H:%M:%S')
        save_dict = {
            "epoch"                : epoch,
            "model_state_dict"     : self.model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "losses"               : losses,
            "MSE_loss"             : MSE_losses, 
            "CEL_loss"             : CEL_losses,
            "timestamp"            : timestamp
        }
        
        filename = f"training_checkpoint"
        torch.save(save_dict, os.path.join(self.save_path, filename + '.pth'))

    def save_ema_checkpoint(self, ema):
        filename = f"EMA_checkpoint"
        torch.save(ema.ema_model.state_dict(), os.path.join(self.save_path, filename + '.pth'))

    def make_config(self, num_epochs, lr, clip, batch_size, path, fixed_N, mean, std):
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
                "optimizer"    : "adam",
                "sample_mean"  : [float(m) for m in mean],
                "sample_std"   : [float(s) for s in std],
                "fixed_N"      : fixed_N
            },
            "path": {
                "name"    :self.path.get_config(),
                "p_simple":self.path.p_simple.get_config(),
                "p_data"  : self.path.p_data.get_config()
            }

        }
        return config

    def save_config_file(self, num_epochs, lr, clip, batch_size, path, fixed_N, mean, std):
        """
        Save yaml with training and model settings.
        """
        config = self.make_config(num_epochs, lr, clip, batch_size, path, fixed_N, mean, std)
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

class ConditionalFlowMatchingTrainer(Trainer):

    """
    Trainer for unguided flow matching.
    """
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module):
        super().__init__(model)
        self.path = path

    def get_train_loss(self, device, batch_size: int) -> torch.Tensor:
        # samples
        z_batch = self.path.p_data.sample(batch_size) # z ~ p_data
        t_batch = torch.rand(batch_size, 1, device=device) # t ~ U(0, 1)

        x0_batch = self.path.p_simple.sample(batch_size)
        x_batch = self.path.sample_conditional_path(x0_batch, z_batch, t_batch) # x ~ p(x|z)

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
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module):
        super().__init__(model)
        self.path = path

    def prepare_batches(self, device, batch_size, **kwargs):
        # samples
        z_batch, y_batch, Ns = self.path.p_data.sample(batch_size) # z, y ~ p_data
        x0_batch = self.path.p_simple.sample(batch_size, Ns=Ns) # x_0 ~ p_simple
        # t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1) 

        # t ~ t^(1/1+a) (inverse sampling)
        u = torch.rand(batch_size, 1, device=device)
        alpha = -0.25
        power = (1 + alpha) / (2 + alpha)
        t_batch = torch.pow(u, power)

        # standardize targets
        z_batch = self.scale_input([z_batch], **kwargs)[0]

        # interpolate initial state and standardized target
        x_batch = self.path.sample_conditional_path(x0_batch, z_batch, t_batch) # x ~ p(x|z)
        
        # scale lightcurve
        AVG_AMP = 150
        y_batch = y_batch / AVG_AMP

        return x0_batch, x_batch, z_batch, t_batch, y_batch, Ns

    def MSE(self, differences):
        squared_error = differences ** 2
        return torch.mean(squared_error)
    
    def get_train_loss(self, device: torch.device, batch_size: int, **kwargs) -> torch.Tensor:
        
        batches = self.prepare_batches(device, batch_size, **kwargs)
        x0_batch, x_batch, z_batch, t_batch, y_batch, Ns = batches

        predicted_field = self.model(x_batch, t_batch, y_batch, Ns)
        target_field    = self.path.conditional_vector_field(x0_batch, z_batch)

        differences = predicted_field - target_field # shape batch_size, ndim
    
        return self.MSE(differences)
    
    def scale_input(self, inputs, mean, std):
        """
        scale inference parameters linearly by lambda.
        """
        for i, input in enumerate(inputs):
            inputs[i] = (input - mean) / std
        return inputs

class TransdimensionalTrainer(GuidedConditionalFlowMatchingTrainer):

    def __init__(self, path: ConditionalProbabilityPath, model: TransdimensionalModel):
        super().__init__(path, model)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.N_classifier = model.component_classifier
        self.vector_field = model.vector_field_model

    def prepare_batches(self, device, batch_size, **kwargs):
        # samples
        z_batch, y_batch, N_batch = self.path.p_data.sample(batch_size) # z, y ~ p_data
        x0_batch = self.path.p_simple.sample(batch_size, Ns=N_batch) # x_0 ~ p_simple
        # t_batch = torch.rand(batch_size, 1) # t ~ U(0, 1) 

        # t ~ t^(1/1+a) (inverse sampling)
        u = torch.rand(batch_size, 1, device=device)
        alpha = -0.25
        power = (1 + alpha) / (2 + alpha)
        t_batch = torch.pow(u, power)

        # standardize targets
        z_batch = self.scale_input([z_batch], **kwargs)[0]

        # interpolate initial state and standardized target
        x_batch = self.path.sample_conditional_path(x0_batch, z_batch, t_batch) # x ~ p(x|z)
        
        # scale lightcurve
        AVG_AMP = 150
        y_batch = y_batch / AVG_AMP 

        return x0_batch, x_batch, z_batch, t_batch, y_batch, N_batch
    
    def get_train_loss(self, device: torch.device, batch_size: int, **kwargs) -> torch.Tensor:
        
        batches = self.prepare_batches(device, batch_size, **kwargs)
        x0_batch, x_batch, z_batch, t_batch, y_batch, N_true = batches

        # N ~ p(N|y)
        N_logits = self.N_classifier(y_batch) # (bs, N_max)
        # p_N = torch.softmax(N_logits, dim=1)
        # N_pred = torch.multinomial(p_N, num_samples=1) + 1 # (bs, 1)

        predicted_field = self.vector_field(x_batch, t_batch, y_batch, N_true)
        target_field    = self.path.conditional_vector_field(x0_batch, z_batch)

        # don't count meaningless tokens
        N_max = N_logits.shape[-1]
        dim = predicted_field.shape[-1]
        N_params = dim // N_max
        mask = torch.arange(N_max, device=N_logits.device).expand(N_logits.shape) < N_true
        
        # this assumes a vector of shape [t0_1, amp_1,..., t0_n, amp_n]
        mask = torch.repeat_interleave(mask, repeats=N_params, dim=1)
        
        # but mask is applied to vector of form [t0_1...t0_n, ..., skew_1...skew_n]
        mask = mask.reshape(batch_size, N_max, N_params).transpose(1, 2).reshape(batch_size, dim)

        # vector field loss
        differences = predicted_field - target_field # shape batch_size, ndim
        MSE = self.MSE(differences[mask])

        # classifier loss
        cross_entropy_loss = self.cross_entropy(N_logits, (N_true - 1).view(-1))
        return MSE, cross_entropy_loss 
    
    def make_config(self, num_epochs, lr, clip, batch_size, path, fixed_N, mean, std):
        try:
            t_encoder_config = self.vector_field.time_seq_encoder.get_config()
        except AttributeError:
            t_encoder_config = False

        config = {
            "model"           : self.vector_field.get_config(),
            "time_seq_encoder": t_encoder_config,
            "theta_encoder"   : self.vector_field.encode_theta,
            "tau_encoder"     : self.vector_field.encode_tau,
            "training":
            {
                "num_epochs"   : num_epochs,
                "learning_rate": lr,
                "batch_size"   : batch_size,
                "gradient_clip": clip,
                "optimizer"    : "adam",
                "sample_mean"  : [float(m) for m in mean],
                "sample_std"   : [float(s) for s in std],
                "fixed_N"      : fixed_N

            },
            "path": {
                "name"    :self.path.get_config(),
                "p_simple":self.path.p_simple.get_config(),
                "p_data"  : self.path.p_data.get_config()
            },
            "classifier": self.N_classifier.get_config()
        }
        return config
