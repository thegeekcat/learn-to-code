# Import modules
import os
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Create a CustomImageDataset class
class CustomImageDataset(Dataset):
    # Initialize the class
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(os.path.join(image_paths, '*', '*.png'))
        self.transform = transform

        self.label_dict = {'australian magpie': 1,
                           'eurasian magpie': 2,
                           'kookaburra': 3,
                           'magpie lark': 4,
                           'myna': 5,
                           'noisy miner': 6}
    
    # Get a single item from the dataset
    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        #print('Image path: ', image_path)
        image = Image.open(image_path)
        #print(image.size)

        folder_name = image_path.split('\\')
        #print('Folder name(Full path):', folder_name)
        folder_name = folder_name[1]
        #print('Folder name: ', folder_name)

        label = self.label_dict[folder_name]
        #print(label)
        
        if self.transform:
            image = self.transform(image)

        #return
        return image, label


    def __len__(self):
        return len(self.image_paths)


# Set image paths
image_paths = './data/birds/'
dataset = CustomImageDataset(image_paths, transform = None)


# Debugging
for image, label in dataset:
    print('Data and Labels: ', image, label)
    #pass





