import torch
print(torch.cuda.is_available())  # Phải trả về True
print(torch.cuda.get_device_name(0))  # Phải trả về "NVIDIA RTX 3070Ti"