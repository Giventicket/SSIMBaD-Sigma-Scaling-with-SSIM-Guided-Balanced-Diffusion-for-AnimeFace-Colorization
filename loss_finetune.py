import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from transformers import CLIPModel, CLIPProcessor

class LossFactory(nn.Module):
    def __init__(self, strategy, model=None):
        super().__init__()
        self.strategy = strategy
        self.model = model
        self.build_losses()

    def build_losses(self):
        if self.strategy in ["mse"]:
            self.mse_loss = nn.MSELoss()

        if self.strategy in ["lpips-vgg"]:
            self.lpips_loss = lpips.LPIPS(net='vgg').eval()
            self.lpips_loss.requires_grad_(False)

        if self.strategy in ["clip"]:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            for p in self.clip_model.parameters():
                p.requires_grad_(False)

    def to(self, device):
        super().to(device)
        self.device = device
        if hasattr(self, 'lpips_loss'):
            self.lpips_loss.to(device)
        if hasattr(self, 'clip_model'):
            self.clip_model.to(device)
        if hasattr(self, 'dino'):
            self.dino.to(device)
        return self

    def preprocess_image(self, x):
        return ((x + 1) / 2).clamp(0, 1)

    def normalize_for_clip(self, x):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
        return (x - mean) / std

    def resize_image(self, x, size=(224, 224), mode="bicubic"):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)

    def safe_normalize(self, feat, eps=1e-6):
        return feat / (feat.norm(dim=-1, keepdim=True) + eps)

    def forward(self, x_gen, x_ref):
        x_gen = self.preprocess_image(x_gen)
        x_ref = self.preprocess_image(x_ref)

        if self.strategy == "lpips-vgg":
            return self.lpips_loss(x_gen, x_ref).mean()

        elif self.strategy == "clip":
            x_gen = self.normalize_for_clip(self.resize_image(x_gen))
            x_ref = self.normalize_for_clip(self.resize_image(x_ref))
            feat_gen = self.clip_model.get_image_features(pixel_values=x_gen)
            feat_ref = self.clip_model.get_image_features(pixel_values=x_ref)
            feat_gen = self.safe_normalize(feat_gen)
            feat_ref = self.safe_normalize(feat_ref)
            return 1.0 - (feat_gen * feat_ref).sum(dim=-1).mean()

        elif self.strategy == "mse":
            return self.mse_loss(x_gen, x_ref)