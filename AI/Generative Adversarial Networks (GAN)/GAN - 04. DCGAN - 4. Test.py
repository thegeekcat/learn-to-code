# Import modules
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from ref_0816_cat_dcgan_model import BaseDcganGenerator


# Set parameters
nz = 100

# Set paths
output_images_path = './outcomes/0816-Cat_DCGAN_generated_images/'
os.makedirs(output_images_path, exist_ok=True)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load trained model
net_generator = BaseDcganGenerator().to(device)
checkpoint_path = ''