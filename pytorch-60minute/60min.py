import numpy as np
import torch

x = torch.rand(5,3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu",torch.double))