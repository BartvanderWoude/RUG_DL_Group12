import modules as md

import torch
import torchvision.utils as vutils
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt


# class DiffusionUNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass

#     def forward(self):
#         pass


class ForwardProcess():
    def __init__(self, T):
        self.T = T
        self.beta = torch.range(start=0.001, end=0.02, step=(0.02-0.001)/T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def __call__(self, x0: torch.Tensor, t):
        noise = torch.sqrt(1-self.alpha_bar[t])*torch.randn(size=x0.shape)
        x_noise = torch.sqrt(self.alpha_bar[t])*x0 + noise
        return x_noise, noise


if __name__ == "__main__":
    x = ForwardProcess(1000)
    print(x.alpha)
    print(x.alpha_bar)

    # input = torch.ones(3, 10, 10)
    # input = input / 2
    input = read_image("puppy.jpg")
    input = input / 255
    x_noise, noise = x(input, 500)
    vutils.save_image(x_noise, "x_noise.png")
    vutils.save_image(noise, "noise.png")

    print(x_noise.shape)
    print(noise.shape)
    print(torch.max(noise))
    print(torch.min(noise))
