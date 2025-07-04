import os
import cv2
import torch
import piq
from torchvision import transforms
from tqdm import tqdm

real_dir = '/data/Anime/test_data/reference'
generated_dir = './result_diff_guided_ssim'

real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
generated_files = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

to_tensor = transforms.Compose([
    transforms.ToTensor(),  # [H, W, C] -> [C, H, W], float32 [0, 1]
])

ms_ssim_scores = []
psnr_scores = []

for real_file, gen_file in tqdm(zip(real_files, generated_files), total=len(real_files)):
    real_path = os.path.join(real_dir, real_file)
    gen_path = os.path.join(generated_dir, gen_file)

    img_real = cv2.imread(real_path)
    img_gen = cv2.imread(gen_path)

    img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))

    img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

    # real_tensor = to_tensor(img_real).unsqueeze(0).cuda()
    # gen_tensor = to_tensor(img_gen).unsqueeze(0).cuda()
    real_tensor = to_tensor(img_real).unsqueeze(0)
    gen_tensor = to_tensor(img_gen).unsqueeze(0)

    ms_ssim = piq.multi_scale_ssim(gen_tensor, real_tensor, data_range=1.0).item()
    ms_ssim_scores.append(ms_ssim)

    psnr = piq.psnr(gen_tensor, real_tensor, data_range=1.0).item()
    psnr_scores.append(psnr)

    print(f"{real_file} vs {gen_file} ➔ MS-SSIM: {ms_ssim:.4f} | PSNR: {psnr:.2f} dB")

avg_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
avg_psnr = sum(psnr_scores) / len(psnr_scores)

print(f"\n📊 Mean MS-SSIM Score: {avg_ms_ssim:.4f}")
print(f"📊 Mean PSNR Score: {avg_psnr:.2f} dB")
