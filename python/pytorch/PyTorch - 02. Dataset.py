# Import modules
import os
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Create a CsutomImageDataset class
class CustomImageDataset(Dataset):

    # Initialize the class
    def __init__(self, image_paths, transform = None):
        # Set image paths
        self.image_paths = glob.glob(os.path.join(image_paths, '*', '*.jpg'))
        # Set transform
        self.transform = transform

        # Set a label dictionary
        self.label_dict = {'dew': 0,
                           'fogsmog': 1,
                           'frost': 2,
                           'glaze': 3,
                           'hail': 4,
                           'lightning': 5,
                           'rain': 6,
                           'rainbow': 7,
                           'rime': 8,
                           'sandstorm': 9,
                           'snow': 10}



    # Get a single item from the dataset
    def __getitem__(self, index):
        # Set image path
        image_path = self.image_paths[index]
        print('Image path: ', image_path)

        # Load image
        image = Image.open(image_path)  # e.g. ./data/0619train/dew/2208.jpg
        #image = image.open(image_path).convert('RGB')

        
        # Get a folder name from paths
        folder_name = image_path.split('\\')  # folder_name = ['./data/0619train', 'dew', '2208.jpg']
        #print('Folder name(Full paths): ', folder_name)
        folder_name = folder_name[1]   # folder_name[1] = 'dew'
        #print('Folder name: ', folder_name)

        # Set a label
        label = self.label_dict[folder_name]
        print(label)

        # Transform image
        if self.transform:
            image = self.transform(image)
        
        return image, label

    # Get the number of items in the dataset    
    def __len__(self):
        return len(self.image_paths)
    

# Set image paths
image_paths = './data/0619train'
dataset = CustomImageDataset(image_paths, transform = None)


# Debugging
for image, label in dataset:
    print('Data and Labels: ', image, label)
    #pass

