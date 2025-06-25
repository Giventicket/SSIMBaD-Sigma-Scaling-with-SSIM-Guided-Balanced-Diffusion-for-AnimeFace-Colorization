import argparse
from pytorch_fid import fid_score
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FID score between real and generated image folders.")
    parser.add_argument("--real_dir", type=str, default="/data/Anime/test_data/reference", help="Directory of reference images.")
    parser.add_argument("--generated_dir", type=str, default="./result_same_finetuned", help="Directory of generated images.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for FID calculation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use: cuda or cpu.")
    parser.add_argument("--dims", type=int, default=2048, help="Feature dimensions (default: 2048 for InceptionV3).")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available. Switching to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    fid_value = fid_score.calculate_fid_given_paths(
        [args.real_dir, args.generated_dir],
        batch_size=args.batch_size,
        device=device,
        dims=args.dims
    )

    print(f"✅ FID Score (reference vs generated): {fid_value:.4f}")

if __name__ == "__main__":
    main()
