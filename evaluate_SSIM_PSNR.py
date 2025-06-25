import os
import cv2
import torch
import piq
from torchvision import transforms
from tqdm import tqdm

# 경로 설정
real_dir = '/data/Anime/test_data/reference'
generated_dir = './result_diff_guided_ssim'

# 이미지 리스트 정렬
real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
generated_files = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Transform 정의
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

    # 크기 맞추기
    img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))

    # BGR to RGB
    img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

    # Tensor 변환 후 배치 차원 추가
    # real_tensor = to_tensor(img_real).unsqueeze(0).cuda()
    # gen_tensor = to_tensor(img_gen).unsqueeze(0).cuda()
    real_tensor = to_tensor(img_real).unsqueeze(0)
    gen_tensor = to_tensor(img_gen).unsqueeze(0)

    # MS-SSIM 계산
    ms_ssim = piq.multi_scale_ssim(gen_tensor, real_tensor, data_range=1.0).item()
    ms_ssim_scores.append(ms_ssim)

    # PSNR 계산
    psnr = piq.psnr(gen_tensor, real_tensor, data_range=1.0).item()
    psnr_scores.append(psnr)

    print(f"{real_file} vs {gen_file} ➔ MS-SSIM: {ms_ssim:.4f} | PSNR: {psnr:.2f} dB")

# 평균 점수 출력
avg_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
avg_psnr = sum(psnr_scores) / len(psnr_scores)

print(f"\n📊 평균 MS-SSIM Score: {avg_ms_ssim:.4f}")
print(f"📊 평균 PSNR Score: {avg_psnr:.2f} dB")
