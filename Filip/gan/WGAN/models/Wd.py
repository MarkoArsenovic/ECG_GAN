import torch
from torch import nn 



class Discriminator1024(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv1d(in_channels=1024, out_channels=channels, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x).view(-1, 1, 256)


class Discriminator512(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)

            #nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=0,bias=False),
            #nn.BatchNorm1d(num_features=16),
            #nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            # output of main module --> State (1024x4x4)
            # State (512x8x8)
            nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm1d(num_features=2048),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)


        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv1d(in_channels=2048, out_channels=channels, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        #print(x.shape)
        x = self.main_module(x)
        #print(x.shape)
        #x = self.output(x)
        #print(x.shape)
        return self.output(x)#.view(-1, 1, 256)

