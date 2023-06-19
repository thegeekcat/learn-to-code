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
        

    # Get a single image from image paths
    def __getitem__(self, index):
        # Get a image path of a single image
        image_path = self.image_paths[index]

        # Open images and convert to RGB
        image = Image.open(image_path).convert('RGB')


        # Check grayscale images
        # Error if RGB channels are different: Grayscale = 1, RGB = 3
        if not is_grayscale(image):
            # Get full paths of the images
            folder_name = image_path.split('\\')
            #print('Folder name(Full path):', folder_name)

            # Get folder names
            folder_name = folder_name[1]
            #print('Folder Name: ', folder_name)


            # Get labels
            label = self.label_dict[folder_name]
            #print('Label: ', label)


            # Transform images
            if self.transform:
                image = self.transform(image)

            # Display result messages
            print('Grayscale images are not detected!')    

            return image, label

        else:
            print('Grayscale images are detected: ', image_path)
    

    def __len__(self):
        return len(self.image_paths)
    
# Transform images
transform = transforms.Compose([
    # Resize images to 224x224
    transforms.Resize((224, 224)),

    # Convert to tensor
    transforms.ToTensor()
])

# Set image paths
image_paths = './data/0619train/'
dataset = CustomImageDataset(image_paths, transform = transform)

# Debugging
for i in dataset:
    #print(i)
    pass
