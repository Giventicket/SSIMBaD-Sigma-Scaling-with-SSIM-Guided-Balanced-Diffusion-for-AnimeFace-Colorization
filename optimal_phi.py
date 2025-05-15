# ─────────────────────────────────────────────────────────────
# 1. Imports & Helper Functions
# ─────────────────────────────────────────────────────────────
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as compare_ssim
from scipy.special import erf, erfinv
from tqdm import tqdm
import random

def get_sigma_samples_from_scaled_space(
    scale_type: str,
    sigma_min: float,
    sigma_max: float,
    steps: int = 50,
    extra_param: float = None
):
    epsilon = 1e-6  # for numerical stability

    if scale_type == "σ":
        phi = lambda σ: σ
        phi_inv = lambda s: s
    elif scale_type == "log":
        phi, phi_inv = np.log, np.exp
    elif scale_type == "log1p":
        phi, phi_inv = np.log1p, np.expm1
    elif scale_type == "σ²":
        phi, phi_inv = lambda σ: σ**2, np.sqrt
    elif scale_type == "1/σ":
        phi, phi_inv = lambda σ: 1/σ, lambda s: 1/s
    elif scale_type == "1/σ²":
        phi, phi_inv = lambda σ: 1/(σ**2), lambda s: 1/np.sqrt(s)
    elif scale_type == "arcsinh":
        phi, phi_inv = np.arcsinh, np.sinh
    elif scale_type == "tanh":
        phi = np.tanh
        phi_inv = lambda s: np.arctanh(np.clip(s, -1 + epsilon, 1 - epsilon))
    elif scale_type == "sigmoid":
        phi = lambda σ: 1 / (1 + np.exp(-σ))
        phi_inv = lambda s: -np.log(1 / np.clip(s, epsilon, 1 - epsilon) - 1)
    elif scale_type == "σ / (σ + c)":
        c = extra_param if extra_param else 0.5
        phi = lambda σ: σ / (σ + c)
        phi_inv = lambda s: c * s / (1 - s)
    elif scale_type == "σᵖ / (σᵖ + 1)":
        p = extra_param if extra_param else 1.5
        phi = lambda σ: (σ ** p) / (σ ** p + 1)
        phi_inv = lambda s: (s / (1 - s)) ** (1 / p)
    elif scale_type == "log(σ² + 1)":
        phi = lambda σ: np.log1p(σ ** 2)
        phi_inv = lambda s: np.sqrt(np.expm1(s))
    elif scale_type == "atan":
        phi, phi_inv = np.arctan, np.tan
    elif scale_type == "erf":
        phi = lambda σ: erf(σ / 5)
        phi_inv = lambda s: 5 * np.sqrt(2) * erfinv(np.clip(s, -1 + epsilon, 1 - epsilon))
    else:
        raise ValueError(f"Unsupported scale_type: {scale_type}")
    
    s_min = phi(sigma_min)
    s_max = phi(sigma_max)
    s_samples = np.linspace(s_min, s_max, steps)
    return phi_inv(s_samples), s_samples

def compute_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return compare_ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

def compute_r2(x, y):
    x = np.array(x)
    y = np.array(y)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    coeffs = np.polyfit(x, y, deg=1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    return 1 - (ss_res / ss_tot)

# ─────────────────────────────────────────────────────────────
# 2. Scaling Configuration
# ─────────────────────────────────────────────────────────────
sigma_min = 0.002
sigma_max = 80
steps = 50

scaling_configs = {
    "σ": lambda: get_sigma_samples_from_scaled_space("σ", sigma_min, sigma_max, steps),
    "log(σ)": lambda: get_sigma_samples_from_scaled_space("log", sigma_min, sigma_max, steps),
    "log1p(σ)": lambda: get_sigma_samples_from_scaled_space("log1p", sigma_min, sigma_max, steps),
    "σ²": lambda: get_sigma_samples_from_scaled_space("σ²", sigma_min, sigma_max, steps),
    "1/σ": lambda: get_sigma_samples_from_scaled_space("1/σ", sigma_min, sigma_max, steps),
    "1/σ²": lambda: get_sigma_samples_from_scaled_space("1/σ²", sigma_min, sigma_max, steps),
    "arcsinh(σ)": lambda: get_sigma_samples_from_scaled_space("arcsinh", sigma_min, sigma_max, steps),
    "tanh(σ)": lambda: get_sigma_samples_from_scaled_space("tanh", sigma_min, sigma_max, steps),
    "sigmoid(σ)": lambda: get_sigma_samples_from_scaled_space("sigmoid", sigma_min, sigma_max, steps),
    "σ / (σ + 0.5)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.5),
    "σᵖ / (σᵖ + 1)": lambda: get_sigma_samples_from_scaled_space("σᵖ / (σᵖ + 1)", sigma_min, sigma_max, steps, extra_param=1.5),
    "log(σ² + 1)": lambda: get_sigma_samples_from_scaled_space("log(σ² + 1)", sigma_min, sigma_max, steps),
    "atan(σ)": lambda: get_sigma_samples_from_scaled_space("atan", sigma_min, sigma_max, steps),
    "σ / (σ + 0.1)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.1),
    "σ / (σ + 0.2)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.2),
    "σ / (σ + 0.3)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.3),
    "σ / (σ + 0.4)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.4),
    "σ / (σ + 0.5)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.5),
    "σ / (σ + 0.6)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.6),
    "σ / (σ + 0.7)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.7),
    "σ / (σ + 0.8)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.8),
    "σ / (σ + 0.9)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=0.9),
    "σ / (σ + 1.0)": lambda: get_sigma_samples_from_scaled_space("σ / (σ + c)", sigma_min, sigma_max, steps, extra_param=1.0)
}

# ─────────────────────────────────────────────────────────────
# 3. Main Evaluation Loop (All Images)
# ─────────────────────────────────────────────────────────────
image_size = 128
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
img_path = '/data/Anime/train_data/reference/'
img_list = sorted(os.listdir(img_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set sample size
# N = len(img_list) // 100  # 최대 100장까지 테스트
N = 20

r2_accumulator = {name: [] for name in scaling_configs.keys()}

shuffled_list = img_list.copy()
random.shuffle(shuffled_list)
subset = shuffled_list[:N]
for filename in tqdm(subset, desc="Processing images"):
    try:
        img = Image.open(os.path.join(img_path, filename)).convert("RGB")
        x0 = transform(img).unsqueeze(0).to(device)
        x0_denorm = (x0[0] * 0.5 + 0.5).clamp(0, 1)

        for name, sampler in scaling_configs.items():
            sigmas, x_scaled = sampler()
            ssim_scores = []

            for sigma in sigmas:
                sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=device)
                noise = torch.randn_like(x0) * sigma_tensor
                xt = (x0 + noise).clamp(-1, 1)
                xt_denorm = (xt[0] * 0.5 + 0.5).clamp(0, 1)

                # SSIM 계산을 CPU로 넘기기 (skimage는 GPU 지원 안함)
                ssim = compute_ssim(x0_denorm.cpu(), xt_denorm.cpu())
                ssim_scores.append(ssim)

            r2 = compute_r2(x_scaled, ssim_scores)
            r2_accumulator[name].append(r2)
            print(name, r2)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 결과 출력
average_r2_results = []
for name, r2_list in r2_accumulator.items():
    if r2_list:
        avg_r2 = np.mean(r2_list)
        average_r2_results.append((name, avg_r2))

average_r2_results.sort(key=lambda x: x[1], reverse=True)

print("\n📊 스케일링 방식별 SSIM vs φ(σ) 선형성 평균 (R² 기준 정렬):")
for name, avg_r2 in average_r2_results:
    print(f"{name:<25} 평균 R² = {avg_r2:.4f}")
