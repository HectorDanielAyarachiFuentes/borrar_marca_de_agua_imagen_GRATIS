import torch

print(f"PyTorch version: {torch.__version__}")
print("-" * 30)

is_cuda = torch.cuda.is_available()
print(f"Is CUDA available? {is_cuda}")

if is_cuda:
    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs found: {gpu_count}")
    for i in range(gpu_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\nCUDA not found. PyTorch is running in CPU-only mode.")
    print("To fix this, you likely need to reinstall PyTorch with CUDA support.")
    print("Visit: https://pytorch.org/get-started/locally/")

