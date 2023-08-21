# Import modules
import torch

# Set parameters for DCGan Model
nz = 100  # Latent vector size
ngf = 64  # Number of Generating Filters: Number of channels in feature maps processed by generator
ndf = 64  # Number of Discriminating Filters: Number of channels in feature maps processed by discriminator
nc = 3    # Number of Color channels -> '3' means RGB


# Set parameters for DCGan Train
data_root = 'G:/My Drive/datasets/0816- Cat Dataset/'
num_workers = 2
batch_size = 128
img_size = 64 # Input image size 64 x 64
num_epochs = 200
lr = 0.00025
beta1 = 0.4 # Adam, AdamW -> bata1 = 0.2  ~ 0.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'