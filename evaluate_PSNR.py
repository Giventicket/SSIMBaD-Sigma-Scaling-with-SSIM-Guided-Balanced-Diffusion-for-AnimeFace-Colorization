import os
import cv2
import torch
import piq
import argparse
from torchvision import transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PSNR between real and generated images.")
    parser.add_argument("--real_dir", type=str, default="/data/Anime/test_data/reference", help="Directory containing reference (real) images.")
    parser.add_argument("--generated_dir", type=str, default="./result_same_finetuned", help="Directory containing generated images.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for computation.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    real_files = sorted([f for f in os.listdir(args.real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    generated_files = sorted([f for f in os.listdir(args.generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(real_files) != len(generated_files):
        print(f"⚠️ File count mismatch: {len(real_files)} real vs {len(generated_files)} generated")
        return

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    psnr_scores = []

    for real_file, gen_file in tqdm(zip(real_files, generated_files), total=len(real_files)):
        real_path = os.path.join(args.real_dir, real_file)
        gen_path = os.path.join(args.generated_dir, gen_file)

        img_real = cv2.imread(real_path)
        img_gen = cv2.imread(gen_path)

        if img_real is None or img_gen is None:
            print(f"⚠️ Failed to load: {real_file} or {gen_file}")
            continue

        img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))
        img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
        img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

        real_tensor = to_tensor(img_real).unsqueeze(0).to(device)
        gen_tensor = to_tensor(img_gen).unsqueeze(0).to(device)

        psnr = piq.psnr(gen_tensor, real_tensor, data_range=1.0).item()
        psnr_scores.append(psnr)

        print(f"{real_file} vs {gen_file} ➔ PSNR: {psnr:.2f} dB")

    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
    print(f"\n📊 Mean PSNR Score: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    main()
