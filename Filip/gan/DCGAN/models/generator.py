import torch.nn as nn

# DC 3 layer generator 
class Generator(nn.Module):
    def __init__(self, ngpu, noise):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose1d(noise, 160, kernel_size = 1, bias=False), # kernel_size = 2
            nn.BatchNorm1d(160),
            nn.ReLU(True),

            nn.ConvTranspose1d(160, 128, kernel_size = 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 160, kernel_size = 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
