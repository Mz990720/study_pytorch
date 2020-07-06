import numpy as np
import torch

x = torch.randn(3,requires_grad=True)
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad:
    print(x.requires_grad)