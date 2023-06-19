# Import modules
import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define a grayscale detector
def is_grayscale(image_grayscale):
    return image_grayscale.mode == "L"


# Create a Custom Image Dataset class
class CustomImageDataset(Dataset):

    # Initialize the class
    def __init__(self, image_paths, transform = None):
        # Set image paths
        self.image_paths = glob.glob(os.path.join(image_paths, '*', '*.png'))
        #print(image_paths)

        # Set transform
        self.transform = transform

        # Set a label dictionary
        self.label_dict = {'australian magpie': 1,
                           'eurasian magpie': 2,
                           'kookaburra': 3,
                           'magpie lark': 4,
                           'myna': 5,
                           'noisy miner': 6}
    
    # Get a single image from the image paths
    def __getitem__(self, index):

        # GEt image paths
        image_path = self.image_paths[index]

        # Open images
        image = Image.open(image_path).convert('RGB')

        # Check grayscale images
        if not is_grayscale(image):

            # Get folder names
            folder_name = image_path.split('\\')
            folder_name = folder_name[1]
            #print(folder_name)

            # Get labels
            label = self.label_dict[folder_name]

            # Transform images
            if self.transform:
                image = self.transform(image)

            # Display a result messsage
            print('Grayscale images are not detected')

            # Return images and labels
            return image, label
        
        else:
            print('Grayscale images are detected: ', image_path)

    
    # Get the length of the dataset
    def __len__(self):
        return len(self.image_paths)

# Transform images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Set image paths
image_paths = './data/birds/'
dataset = CustomImageDataset(image_paths, transform = transform)

# Debugging
for i in dataset:
    #print(i)
    pass





