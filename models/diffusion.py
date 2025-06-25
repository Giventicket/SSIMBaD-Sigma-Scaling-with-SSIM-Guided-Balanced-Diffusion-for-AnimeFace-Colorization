from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.unet import UNet
import lpips
import piq

class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        inference_time_step,
        unet,
        c,
        sigma_min,
        sigma_max,
        sigma_data,
        S_churn,
        S_min,
        S_max,
        S_noise
    ):
        super().__init__()

        # member variables
        self.denoise_fn = UNet(**unet)
        self.inference_time_step = inference_time_step
        self.c = c
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        # parameters
        sigma_min = max(self.S_min, self.sigma_min)
        sigma_max = min(self.S_max, self.sigma_max)
        
        # reflect bound squash sampling concept
        step_indices = torch.arange(self.inference_time_step)
        phi_min = sigma_min / (sigma_min + c)
        phi_max = sigma_max / (sigma_max + c)
        
        # Uniform in φ-space
        phi = phi_min + (phi_max - phi_min) * (step_indices / (self.inference_time_step - 1))
        # Inverse: σ = c·φ / (1 - φ)
        sigma_steps = (c * phi) / (1 - phi)
        sigma_steps = sigma_steps.flip(0)
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])  # t_N = 0
        
        self.register_buffer('inference_time_steps', sigma_steps)
        
        # loss function
        self.loss_fn = partial(F.mse_loss, reduction="none")
        self.lpips_loss_fn = None

    def inference(self, x_t, x_cond=None):
        ret = []
        
        # Main sampling loop.
        x_next = x_t * self.inference_time_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(self.inference_time_steps[:-1], self.inference_time_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.inference_time_step, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            
            # No stochasticity.
            # t_hat = t_cur
            # x_hat = x_cur

            # Euler step.
            denoised = self.forward(x_hat, t_hat.flatten().reshape(-1, 1, 1, 1), x_cond)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.inference_time_step - 1:
                denoised = self.forward(x_next, t_next.flatten().reshape(-1, 1, 1, 1), x_cond)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            ret.append(x_next)
            
        return ret    
    
    def inference_euler_only(self, x_t, x_cond=None, time_steps=50):
        """
        Performs Euler-based inference with a custom number of steps.

        Args:
            x_t (Tensor): Initial noise [B, C, H, W]
            x_cond (Tensor): Conditioning tensor
            time_steps (int): Number of inference steps (effective steps = time_steps + 1)

        Returns:
            List of intermediate denoised outputs
        """
        ret = []

        # Subsample time steps to match requested number
        full_steps = self.inference_time_steps  # [t_0, ..., t_50]
        if time_steps + 1 > len(full_steps):
            raise ValueError(f"Requested time_steps={time_steps} is too large for available steps ({len(full_steps)}).")

        # Uniformly sample time_steps+1 values from full_steps
        idx = torch.linspace(0, len(full_steps) - 1, time_steps + 1).long()
        inference_steps = full_steps[idx]

        # Initial step
        x_next = x_t * inference_steps[0]

        # Euler loop
        for t_cur, t_next in zip(inference_steps[:-1], inference_steps[1:]):
            x_cur = x_next
            t_hat = t_cur
            x_hat = x_cur

            # Euler step
            denoised = self.forward(x_hat, t_hat.flatten().reshape(-1, 1, 1, 1), x_cond)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            ret.append(x_next)

        return ret

    def forward(self, x, sigma, x_cond):
        """
        :param[in]  x       torch.Tensor    [batch_size x channel x height x width]
        :param[in]  sigma       torch.Tensor    [batch_size x 1 x 1 x 1]
        :param[in]  x_cond  torch.Tensor    [batch_size x _ x height x width]
        """
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        
        # reflect bound squash sampling concept
        c_noise = sigma / (sigma + self.c)
        
        F_x = self.denoise_fn(torch.cat([c_in * x, x_cond], dim=1), c_noise.flatten())
        D_x = c_skip * x + c_out * F_x
        
        return D_x
    
    def compute_loss(self, x_ref, D_x):
        """
        :param[in]  x_ref       torch.Tensor    [batch_size x channel x height x width] 
        :param[in]  D_x       torch.Tensor    [batch_size x channel x height x weight]
        :param[in]  weight       torch.Tensor    [batch_size x 1 x 1 x 1]
        """
        
        return self.loss_fn(D_x, x_ref).mean()

    # lpips => 10, ssim => 100
    def inference_guided(self, x_t, x_ref, x_cond=None, guidance_scale=500):
        ret = []
        x_next = x_t * self.inference_time_steps[0]
        
        # LPIPS settings
        # if self.lpips_loss_fn is None:
        #     self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(x_t.device)
        #     self.lpips_loss_fn.eval()

        for t_cur, t_next in zip(self.inference_time_steps[:-1], self.inference_time_steps[1:]):
            with torch.cuda.amp.autocast(enabled=False), torch.enable_grad():
                x_hat = x_next.detach().clone().float().requires_grad_(True)
                t_cur_tensor = t_cur.view(-1, 1, 1, 1).to(dtype=torch.float32, device=x_hat.device)
                t_next_tensor = t_next.view(-1, 1, 1, 1).to(dtype=torch.float32, device=x_hat.device)

                x_cond = x_cond.to(dtype=torch.float32, device=x_hat.device)
                x_ref_clean = x_ref.detach().clone().to(dtype=torch.float32, device=x_hat.device)

                denoised = self.forward(x_hat, t_cur_tensor, x_cond)
                d_cur = (x_hat - denoised) / (t_cur_tensor + 1e-8)
                x_euler = x_hat + (t_next_tensor - t_cur_tensor) * d_cur

                # LPIPS는 입력이 [0, 1]로 정규화된 RGB 이미지여야 함
                x_euler_norm = (x_euler.clamp(-1, 1) + 1) / 2
                x_ref_norm = (x_ref_clean.clamp(-1, 1) + 1) / 2
                
                x_euler_gray = x_euler_norm.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                x_ref_gray = x_ref_norm.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

                # Lpips Loss
                # loss = self.lpips_loss_fn(x_euler_gray, x_ref_gray).mean()
                
                # ssim Loss
                ssim_val = piq.ssim(x_euler_gray, x_ref_gray, data_range=1.0)
                loss = 1 - ssim_val

                grad = torch.autograd.grad(loss, x_hat, retain_graph=False)[0]

            x_next = (x_euler - guidance_scale * grad).detach()
            ret.append(x_next)

        return ret
    