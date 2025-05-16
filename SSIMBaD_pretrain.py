import os
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.anime_train import Anime
from models.diffusion import GaussianDiffusion
import utils.image
import utils.path
import torch.nn.utils as nn_utils

class SSIMBaD(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.save_hyperparameters(cfg)
        
        unet = {
            "channel_in": self.cfg.channel_in,
            "channel_out": self.cfg.channel_out,
            "channel_mult": self.cfg.channel_mult,
            "attention_head": self.cfg.attention_head,
            "cbam": self.cfg.cbam,
        }
        
        self.sigma_data = self.cfg.sigma_data
        
        self.model = GaussianDiffusion(
            inference_time_step=cfg.inference_time_step,
            unet=unet,
            c = cfg.c,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            S_churn=cfg.S_churn,
            S_min=cfg.S_min,
            S_max=cfg.S_max,
            S_noise=cfg.S_noise,
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr * len(self.cfg.gpus),
            weight_decay=self.cfg.weight_decay
        )

        def lr_lambda(step):
            epoch = (step / len(self.train_dataloader()) + self.current_epoch)
            
            # Warmup
            if epoch < self.cfg.warmup_epochs:
                warmup_ratio = epoch / self.cfg.warmup_epochs
                return warmup_ratio * (1 - self.cfg.min_lr/self.cfg.lr) + self.cfg.min_lr/self.cfg.lr

            # Cosine Decay
            progress = (epoch - self.cfg.warmup_epochs) / (self.cfg.epochs - self.cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - self.cfg.min_lr/self.cfg.lr) + self.cfg.min_lr/self.cfg.lr

        # Scheduler
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        self.train_dataset = Anime(
            reference_path = self.cfg.train_reference_path, 
            condition_path = self.cfg.train_condition_path,
            size = self.cfg.size,
        )
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            num_workers=64,
            prefetch_factor=1
        )
        return train_dataloader

    def test_dataloader(self):
        self.test_dataset = Anime(
            reference_path = self.cfg.test_reference_path, 
            condition_path = self.cfg.test_condition_path,
            size = self.cfg.size,
        )
        test_dataset = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            pin_memory=True,
            drop_last=True,
        )
        return test_dataset
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"Total Parameters: {total_params:,}")
        self.print(f"Trainable Parameters: {trainable_params:,}")

    def pretraining_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]
        
        # [B, 1, H, W] + [B, 3, H, W] → [B, 4, H, W]
        x_cond = torch.cat([x_con, x_dis], dim=1)
        
        # sampling from φ(σ) = σ / (σ + c), where c = 0.3 is the best
        c = self.cfg.c
        rnd_uniform = torch.rand([x_ref.shape[0], 1, 1, 1], device=self.device)
        # compute φ(σ_min), φ(σ_max)
        phi_min = self.cfg.sigma_min / (self.cfg.sigma_min + c)
        phi_max = self.cfg.sigma_max / (self.cfg.sigma_max + c)
        # uniform sampling in φ-space
        phi = rnd_uniform * (phi_max - phi_min) + phi_min
        # invert φ to get σ: σ = c·φ / (1 - φ)
        sigma = (c * phi) / (1 - phi)
        
        noise = torch.randn_like(x_ref) * sigma
        D_x = self.model(x_ref + noise, sigma, x_cond)
        loss = self.model.compute_loss(x_ref, D_x)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, logger=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.pretraining_step(batch, batch_idx)
            
    def on_train_epoch_end(self):
        avg_loss = self.all_gather(self.trainer.callback_metrics["train_loss"]).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)
        
        if self.trainer.is_global_zero:
            self.print(f"Epoch {self.current_epoch} - Avg Loss: {avg_loss:.4f}")
            
    def test_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]

        noise = torch.randn_like(x_ref).to(self.device)  # [B, 3, H, W]

        
        with torch.no_grad():
            rets = self.model.inference(
                x_t=noise,
                x_cond=torch.cat([x_con, x_dis], dim=1),
            )[-1]

        images = utils.image.tensor2PIL(rets)
        for i, filename in enumerate(batch['name']):
            output_path = os.path.join(self.cfg.test_output_dir, f'ret_{filename}')
            images[i].save(output_path)
        
        """    
        with torch.no_grad():
            rets = self.model.inference(
                x_t=noise,
                x_cond=torch.cat([x_con, x_dis], dim=1),
            )  # List[Tensor], each: [B, 3, H, W]

        for i, filename in enumerate(batch['name']):
            # Create sub-folder for each filename
            image_dir = os.path.join(self.cfg.test_output_dir, filename)
            os.makedirs(image_dir, exist_ok=True)
                
            for step_idx, img_tensor in enumerate(rets):  # [B, 3, H, W]
                images = utils.image.tensor2PIL(img_tensor)  # returns list of PIL images
                for i, filename in enumerate(batch['name']):
                    image_path = os.path.join(self.cfg.test_output_dir, filename, f"step_{step_idx}.png")
                    images[i].save(image_path)
        """


    def on_test_epoch_end(self):
        self.print(f"All test outputs saved to {self.cfg.test_output_dir}")
