# scripts/test_preprocess.py

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipetrt.landmark.preprocess import preprocess_image


IMAGE_PATH = PROJECT_ROOT / "samples" / "hand.jpg"


def main():
    input_tensor = preprocess_image(IMAGE_PATH)

    print("\n=== PREPROCESS RESULT ===")
    print(f"shape : {input_tensor.shape}")
    print(f"dtype : {input_tensor.dtype}")
    print(f"range : {input_tensor.min()} ～ {input_tensor.max()}")
    print(f"連続配置 : {input_tensor.flags['C_CONTIGUOUS']}")


if __name__ == "__main__":
    main()