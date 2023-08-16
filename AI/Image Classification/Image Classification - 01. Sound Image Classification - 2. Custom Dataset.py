# Import modules
import os
from torch.utils.data import Dataset
from PIL import Image
import glob

# Define a class
class CustomDataset(Dataset):
    # Initialize the class
    def __init__(self, data_dir, transform=None):
        # 'data_dir': ./data/0713-Sound Images/train/
        self.data_dir = glob.glob(os.path.join(data_dir, '*', '*.png'))
        self.transform = transform

    # Define 'Get image files'
    def __getitem__(self, item):
        image_path = self.data_dir[item]
        print(image_path)
        image = Image.open(image_path)

        return image


    # Define 'Check Lengths'
    def __len__(self):
        return len(self.data_dir)

test = CustomDataset('./data/0713-Sound Images/train', transform=None)

for i in test:
    pass


