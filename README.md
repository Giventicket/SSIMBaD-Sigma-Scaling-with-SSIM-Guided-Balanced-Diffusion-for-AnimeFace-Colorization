# SSIMBaD: Sigma Scaling with SSIM-Guided Balanced Diffusion for AnimeFace Colorization

> Official PyTorch implementation of
> **"Sigma Scaling with SSIM-Guided Balanced Diffusion for AnimeFace Colorization"**


![ssimbad](https://github.com/user-attachments/assets/fc8d13d2-7fde-49dd-972c-1f24daa8c5c5)


## 1. Overview

**SSIMBaD** introduces a novel diffusion-based framework for automatic colorization of anime-style facial sketches. Unlike prior DDPM/EDM-based methods that rely on handcrafted or fixed noise schedules, SSIMBaD leverages a perceptual noise schedule grounded in **SSIM-aligned sigma-space scaling**. This design enforces **uniform perceptual degradation** throughout the diffusion process, improving both **structural fidelity** and **stylistic accuracy** in the generated outputs.

The following table compares baseline models and our proposed SSIMBaD framework under both **same-reference** and **cross-reference** settings. Metrics include **PSNR** (higher is better), **MS-SSIM** (higher is better), and **FID** (lower is better).

| Method                                    | Training         | PSNR ↑ (Same / Cross) | MS-SSIM ↑ (Same / Cross) | FID ↓ (Same / Cross) |
|-------------------------------------------|------------------|------------------------|---------------------------|-----------------------|
| SCFT [Lee2020]                            | 300 epochs       | 17.17 / 15.47          | 0.7833 / 0.7627           | 43.98 / 45.18         |
| AnimeDiffusion (pretrained) [Cao2024]     | 300 epochs       | 11.39 / 11.39          | 0.6748 / 0.6721           | 46.96 / 46.72         |
| AnimeDiffusion (finetuned) [Cao2024]      | 300 + 10 epochs  | 13.32 / 12.52          | 0.7001 / 0.5683           | 135.12 / 139.13       |
| SSIMBaD (w/o trajectory refinement)       | 300 epochs       | 15.15 / 13.04          | 0.7115 / 0.6736           | 53.33 / 55.18         |
| **SSIMBaD (w/ trajectory refinement)** 🏆 | **300 + 10 epochs** | **18.92 / 15.84**      | **0.8512 / 0.8207**       | **34.98 / 37.10**


This repository includes:

* 🧠 Pretraining with classifier-free guidance and structural reconstruction loss
* 🎯 Finetuning with perceptual objectives and SSIM-guided trajectory refinement
* 📈 Perceptual noise schedule design based on SSIM curve fitting
* 🧪 Full evaluation pipeline for same-reference and cross-reference scenarios
---

## 2. Key Idea

The quality of diffusion-based generation is highly sensitive to how noise levels are scheduled over time.

Existing models like DDPM and EDM use different schedules for training and inference (e.g., log(σ) vs. σ<sup>1/ρ</sup>), often leading to **perceptual mismatches** that degrade visual consistency.

To resolve this, we introduce a shared transformation ϕ: ℝ<sub>+</sub> → ℝ used **consistently** in both training and generation.  
This transformation maps the raw noise scale σ to a perceptual difficulty axis, allowing uniform degradation in image quality over time.

We select the optimal transformation ϕ\* by maximizing the linearity of SSIM degradation over a candidate set Φ:

<div align="center">
  <img width="667" alt="스크린샷 2025-05-15 오후 5 23 10" src="https://github.com/user-attachments/assets/40acbc96-3208-49a3-b549-7851aa6132c6" />
</div>

This function achieves the best perceptual alignment, leading to **smooth and balanced degradation curves**.

> 🧞‍♂️ Think of ϕ\*(σ) as a perceptual "magic carpet ride" — smooth, stable, and optimally guided by structural similarity.

---

## 3. Folder Overview

```
├── pretrain.py                 # SSIMBaD training (EDM + φ*(σ))
├── finetune.py                 # Trajectory refinement stage
├── AnimeDiffusion_pretrain.py # Baseline reproduction (vanilla EDM schedule)
├── AnimeDiffusion_finetune.py # Baseline finetuning (MSE-based)
├── evaluate_*.py              # FID / PSNR / SSIM evaluation
├── optimal_phi.py             # φ*(σ) search via SSIM R² maximization
├── models/                    # Diffusion & U-Net architectures
├── utils/                     # XDoG, TPS warp, logger, path utils
└── requirements.txt
```
---

## 4. Noise Schedule Analysis

We visualize how SSIM degrades across diffusion timesteps for various noise schedules:

### SSIM Degradation Curves

| DDPM                            | EDM                            | SSIMBaD                        |
| ------------------------------- | ------------------------------ | ------------------------------ |
| ![](assets/ssim_curve_ddpm.png) | ![](assets/ssim_curve_edm.png) | ![](assets/ssim_curve_phi.png) |

### Corresponding Noisy Image Grids

| DDPM                                  | EDM                                  | SSIMBaD                              |
| ------------------------------------- | ------------------------------------ | ------------------------------------ |
| ![](assets/noisy_image_grid_ddpm.png) | ![](assets/noisy_image_grid_edm.png) | ![](assets/noisy_image_grid_phi.png) |


---

## 5. Installation

```bash
conda create -n ssimbad python=3.9
conda activate ssimbad
pip install -r requirements.txt
```

---

## 6. Dataset

* Dataset: **Danbooru Anime Face Dataset**
* Each sample = (`Igt`, `Isketch`, `Iref`)
* Sketch: Generated with XDoG filter
* Reference: TPS + rotation warped version of ground truth

**Prepare like:**

```bash
data/
├── train/
│   ├── 0001_gt.png
│   ├── 0001_sketch.png
│   └── 0001_ref.png
├── val/
...
```

---
## 7. Pretraining

The pretraining stage optimizes the base diffusion model using MSE loss between predicted and ground-truth RGB images, with sketch and reference inputs. It forms the foundation for perceptual finetuning.

```bash
python train_pretrain.py \
    --do_train True \
    --epochs 300 \
    --train_batch_size 32 \
    --inference_time_step 500 \
    --train_reference_path /data/Anime/train_data/reference/ \
    --train_condition_path /data/Anime/train_data/sketch/ \
    --gpus 0, 1
```

---

## 8. Finetuning (trajectory refinement)

This project supports perceptual finetuning using pre-trained diffusion weights.  
You can specify the strategy (e.g., `lpips-vgg`, `clip`, or `mse`) and resume from a checkpoint.

```bash
python train.py \
    --do_train True \
    --resume_from_checkpoint /path/to/checkpoint.ckpt \
    --strategy mse \
    --epochs 10 \
    --finetuning_inference_time_step 50 \
    --train_reference_path /data/Anime/train_data/reference/ \
    --train_condition_path /data/Anime/train_data/sketch/ \
    --gpus 0 1
```
---


## 9. Test

Model evaluation can be performed via either `train.py` (for pretrained models) or `finetune.py` (for finetuned models). Currently, testing is integrated within the training scripts, but we plan to provide a **dedicated and cleaner testing interface** in a future update for ease of use and clarity. Stay tuned for a streamlined `test.py` entry point.

---

## 10. Implementation Details

We implement our training and evaluation pipeline using [PyTorch Lightning](https://www.pytorchlightning.ai/), enabling modular, scalable, and reproducible experimentation.

- **Hardware:** All experiments were conducted on a single node equipped with **2× NVIDIA H100 (80GB)** GPUs.
- **Distributed Training:** We use **DDP (Distributed Data Parallel)** via `strategy="ddp_find_unused_parameters_true"` for efficient multi-GPU training.
- **Mixed Precision:** Enabled via `precision="16-mixed"` to accelerate training and reduce memory usage without sacrificing model quality.
- **Training Epochs:** Pretraining was conducted for 300 epochs using MSE loss.
- **Inference Timesteps:** Default forward process uses 500 steps unless specified otherwise.
- **Checkpointing:** Top-3 checkpoints are saved based on training loss using:
  ```python
  ModelCheckpoint(monitor="train_avg_loss", save_top_k=3, mode="min")

---

## 11. Evaluation

* **FID**:

```bash
python evaluate_FID.py \
```

* **PSNR + SSIM**:

```bash
python evaluate_SSIM_PSNR.py
```
---
