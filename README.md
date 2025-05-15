# SSIMBaD: SSIM-Guided Balanced Diffusion for Anime Face Colorization

> Official implementation of
> **"Sigma Scaling with SSIM-Guided Balanced Diffusion for AnimeFace Colorization"** (NeurIPS 2025 submission)
> Includes pretraining, finetuning, perceptual noise schedule design, and full evaluation

---

## 🧠 Key Idea

The model uses **SSIM-guided sigma scaling**

$$
\phi^*(\sigma) = \frac{\sigma}{\sigma + 0.3}
$$

to ensure perceptual uniformity in both training and generation.
It improves SSIM stability and sample quality compared to DDPM and vanilla EDM.

---

## 📦 Folder Overview

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

## 🚀 Installation

```bash
git clone https://github.com/yourname/SSIMBaD.git
cd SSIMBaD

conda create -n ssimbad python=3.9
conda activate ssimbad
pip install -r requirements.txt
```

---

## 🖼️ Dataset

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

## 🧪 Training SSIMBaD

```bash
python pretrain.py \
  --data_path ./data/train/ \
  --save_path ./checkpoints/ssimbad/ \
  --phi_type sigmoid_custom \
  --use_phi_star True \
  --num_epochs 300
```

*This uses φ\*(σ) = σ / (σ + 0.3)* for both noise sampling and embedding.
Check `optimal_phi.py` to search the best φ empirically.

---

## 🎯 Trajectory Refinement

```bash
python finetune.py \
  --model_path ./checkpoints/ssimbad/epoch300.pth \
  --save_path ./checkpoints/ssimbad_refined/ \
  --num_epochs 10
```

This is NOT a generic MSE finetuning like AnimeDiffusion.
It optimizes the **reverse trajectory** using perceptual noise scaling.

---

## 🧪 Baselines

* **AnimeDiffusion (vanilla EDM):**

```bash
python AnimeDiffusion_pretrain.py
```

* **Finetune AnimeDiffusion:**

```bash
python AnimeDiffusion_finetune.py
```

---

## 📊 Evaluation

* **FID**:

```bash
python evaluate_FID.py \
  --pred_dir ./results/ \
  --gt_dir ./data/val/
```

* **PSNR + SSIM**:

```bash
python evaluate_SSIM_PSNR.py
```

---

## 📈 Noise Schedule Analysis

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

## 📜 License

MIT License