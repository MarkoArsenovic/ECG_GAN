import torch
from torch import nn 


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose1d(in_channels=64, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose1d(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        print(x.shape)
        x = x.view(-1,256)
        print(x.shape)
        return self.output(x)
