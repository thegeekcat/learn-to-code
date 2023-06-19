# Import modules
import json
from PIL import Image
import torch
import os
from torch.utils.data import Dataset 


# Define a class to load JSON file
class CustomDataset(Dataset):

    # Initialize the class
    def __init__(self, json_path, transforms=None):
        # Initialize the transforms
        self.transforms = transforms

        # Load JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)


    # Define the __getitem__ method
    def __getitem__(self, index):

        #print(self.data)
        # Get image paths
        image_path = self.data[index]['filename']
        image_path = os.path.join('./data', image_path)

        # Read images
        #image = Image.open(image_path).convert('RGB')

        # Get coordinates of bounding boxes
        bboxes = self.data[index]['ann']['bboxes']
        labels = self.data[index]['ann']['labels']

        # Change the info of bounding boxes to Tensors
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image_path, {'bboxes': bboxes, 'labels': labels}
    
    # Define the __len__ method
    def __len__(self):
        return len(self.data)

# Set JSON file paths
data = CustomDataset('./data/0619-test_json.json', transforms=None)

# Debugging
for image_paths, anno in data:
    #pass
    print(image_paths, anno['bboxes'], anno['labels'])