"""Run this script to see if you have GPU available"""

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should be >= 1
print(torch.cuda.get_device_name(0))  # Name of the GPU

