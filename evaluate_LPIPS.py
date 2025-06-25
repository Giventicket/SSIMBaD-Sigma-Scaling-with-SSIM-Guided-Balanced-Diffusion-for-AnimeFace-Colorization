import os
import argparse
import torch
import lpips
import cv2
from torchvision import transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS score between real and generated image folders.")
    parser.add_argument("--real_dir", type=str, default="/data/Anime/test_data/reference", help="Directory of reference images.")
    parser.add_argument("--generated_dir", type=str, default="./result_same_finetuned", help="Directory of generated images.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use.")
    parser.add_argument("--net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available. Switching to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    # Initialize LPIPS metric
    loss_fn = lpips.LPIPS(net=args.net).to(device)

    # Collect file names
    real_files = sorted([f for f in os.listdir(args.real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    generated_files = sorted([f for f in os.listdir(args.generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(real_files) != len(generated_files):
        print(f"⚠️ File count mismatch: {len(real_files)} real vs {len(generated_files)} generated")
        return

    to_tensor = transforms.ToTensor()
    lpips_scores = []

    for real_file, gen_file in tqdm(zip(real_files, generated_files), total=len(real_files)):
        real_path = os.path.join(args.real_dir, real_file)
        gen_path = os.path.join(args.generated_dir, gen_file)

        img_real = cv2.imread(real_path)
        img_gen = cv2.imread(gen_path)

        if img_real is None or img_gen is None:
            print(f"⚠️ Failed to load: {real_file} or {gen_file}")
            continue

        # Resize and convert to RGB
        img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))
        img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
        img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize to [-1, 1]
        real_tensor = to_tensor(img_real).unsqueeze(0).to(device) * 2 - 1
        gen_tensor = to_tensor(img_gen).unsqueeze(0).to(device) * 2 - 1

        # Compute LPIPS
        dist = loss_fn(gen_tensor, real_tensor).item()
        lpips_scores.append(dist)

        print(f"{real_file} vs {gen_file} ➔ LPIPS: {dist:.4f}")

    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0.0
    print(f"\n📊 Mean LPIPS Score: {avg_lpips:.4f} (lower is better)")

if __name__ == "__main__":
    main()
