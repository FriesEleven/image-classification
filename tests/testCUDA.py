import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")
if torch.backends.cudnn.is_available():
        print("cuDNN is available.")
else:
        print("cuDNN is not available.")