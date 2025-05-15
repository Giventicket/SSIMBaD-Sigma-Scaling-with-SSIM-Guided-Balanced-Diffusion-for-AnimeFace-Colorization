from pytorch_fid import fid_score
import torch

# 이미지 폴더 경로 설정
images_path = '/data/Anime/test_data/reference'
generated_images_path = './result_same_finetuned'

# GPU 사용 가능 여부 확인
device = 0

# FID 계산
fid_value = fid_score.calculate_fid_given_paths(
    [images_path, generated_images_path],
    batch_size=50,
    device=device,
    dims=2048
)

print(f"✅ sketch = reference FID Score: {fid_value}")
