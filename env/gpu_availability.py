import torch

print(f"torch: {torch.__version__}")
print(f"is cuda available : {torch.cuda.is_available()}")
print(f"gpu count : {torch.cuda.device_count()}")
print(f"device name : {torch.cuda.get_device_name(0)}")
print(f"current device: {torch.cuda.current_device}")