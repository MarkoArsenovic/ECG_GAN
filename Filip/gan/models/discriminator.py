import torch.nn as nn

# TODO make in_channels passed in constructor
# DC 3 layer discriminator 
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(in_channels = 160, out_channels = 128, kernel_size = 1, bias=False),#, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=1 ,bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(512, 1,kernel_size=1, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)

