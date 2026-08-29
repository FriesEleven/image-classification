"""Print the available PyTorch accelerator."""

import torch


def main() -> int:
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        print("Apple MPS available")
    else:
        print("No CUDA/MPS accelerator detected; CPU will be used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
