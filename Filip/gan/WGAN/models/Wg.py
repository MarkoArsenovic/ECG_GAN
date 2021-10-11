import torch
from torch import nn 


class Generator1024(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose1d(in_channels=100, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose1d(in_channels=256, out_channels=channels, kernel_size=4, stride=1, padding=0, bias=True))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        #x = x.view(-1,256)
        return self.output(x)


class Generator512(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            

            nn.ConvTranspose1d(in_channels=100, out_channels=2048, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(in_channels=2048, out_channels=512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(True),
            
            
            # State (256x16x16)
            nn.ConvTranspose1d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1, bias=True))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        #x = x.view(-1,256)
        return self.output(x)

class Thrash():
    
    """
                nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(True),
            
            # State (256x16x16)
            nn.ConvTranspose1d(in_channels=16, out_channels=channels, kernel_size=4, stride=2, padding=1, bias=True))
    """
    pass