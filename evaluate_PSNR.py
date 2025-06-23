import os
import cv2
import torch
import piq
from torchvision import transforms
from tqdm import tqdm

real_dir = '/data/Anime/test_data/reference'
generated_dir = './result_diff'

real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
generated_files = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

psnr_scores = []

for real_file, gen_file in tqdm(zip(real_files, generated_files), total=len(real_files)):
    real_path = os.path.join(real_dir, real_file)
    gen_path = os.path.join(generated_dir, gen_file)

    img_real = cv2.imread(real_path)
    img_gen = cv2.imread(gen_path)

    img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))

    img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

    real_tensor = to_tensor(img_real).unsqueeze(0).cuda()
    gen_tensor = to_tensor(img_gen).unsqueeze(0).cuda()

    psnr = piq.psnr(gen_tensor, real_tensor, data_range=1.0).item()
    psnr_scores.append(psnr)

    print(f"{real_file} vs {gen_file} ➔ PSNR: {psnr:.2f} dB")

avg_psnr = sum(psnr_scores) / len(psnr_scores)
print(f"\n📊 Mean PSNR Score: {avg_psnr:.2f} dB")
