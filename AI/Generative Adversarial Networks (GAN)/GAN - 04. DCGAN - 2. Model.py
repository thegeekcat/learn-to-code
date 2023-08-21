# Import modules
import torch.nn as nn
from ref_0816_cat_dcgan_config import nc, nz, ndf, ngf

"""
nz = 100  # Latent vector size
ngf = 64  # Number of Generating Filters: Number of channels in feature maps processed by generator
ndf = 64  # Number of Discriminating Filters: Number of channels in feature maps processed by discriminator
nc = 3    # Number of Color channels -> '3' means RGB
"""

# Define a class
class BaseDcganGenerator(nn.Module):
    # Initialize the class
    def __init__(self):
        super(BaseDcganGenerator, self).__init__()
        self.main = nn.Sequential(
            # ConvTranspose
            # input: 28x28, nz = 100, ngf = 64
            nn.ConvTranspose2d(nz,       # Num of input channels: nz(Latent vector size )= 100
                               ngf*8,    # Num of output channels: ngf(Number of generating filters) * 8 = 64 * 8 = 512
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(ngf * 8),    # Num of features: ngf * 8 = 64 * 8 = 512
            nn.ReLU(),


            # (ngf x 8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 512, 256
            nn.BatchNorm2d(ngf * 4), # Num of features: ngf * 4 = 64 * 4 = 256
            nn.ReLU(),

            # (ngf x 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 256, 128
            nn.BatchNorm2d(ngf * 2), # Num of features: ngf * 2 = 64 * 2 = 128
            nn.ReLU(),

            # (ngf x 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),  # 128, 64
            nn.BatchNorm2d(ngf), # Num of features: ngf * 1 = 64 * 1 = 64
            nn.ReLU(),

            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False), # 64, 3
            nn.Tanh() # Reason of using 'Tanh()': To match its pixel values between '-1' and '1'
                      #                          as discriminator is designed for images with pixel values between '-1' and '1'
        )

    # Define forward
    def forward(self, input):
        return self.main(input)


# Define a class
class BaseDcganDiscriminator(nn.Module):
    # Initialize the class
    def __init__(self):
        super(BaseDcganDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),  # Default: 0.1 - 0.3

            # (ngf x 2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),

            # (ngf x 4) x 8 x 8
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),

            # (ngf x 8) x 4 x 4
            nn.Conv2d(ndf*4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),

            # Output layer
            nn.Conv2d(ndf * 8, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    # Define forward
    def forward(self, input):
        return self.main(input)










