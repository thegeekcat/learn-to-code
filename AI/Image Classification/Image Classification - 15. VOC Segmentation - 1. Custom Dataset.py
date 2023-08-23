# Import modules
import numpy as np
import os
import cv2
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor

# Define a class for dataset for VOC Segmentation
class CustomDataset(VOCSegmentation):
    # Initialize the class
    def __init__(self, path_dataset, **kwargs):
        #self.createFolder(path_dataset)
        self.path_dataset = path_dataset
        download = self.check_if_path_exists()
        # print(download)
        # exit()
        super().__init__(path_dataset, download=download, **kwargs) # 'super().__init__(path, **kwargs)': Except 'path', all things are inherited from the parent class

    # Define __getitem__ to load images and masks
    def __getitem__(self, index):
        # Read image paths
        #  - Reason of using 'cv2.imread()': As the list contains 'PATH' only instead of image data itself
        img = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)  # As pretrained model in 'DeepLabv3' uses images in Gray scale
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Set augmentations
        # Notice:
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            mask[mask > 20] = 0  # As there are 21 classes


        ####### NEED TO TEST HERE
        #img = to_tensor(img)
        #mask = torch.from_numpy(mask).type(torch,long)
        #print(img, mask)
        return img, mask

    # Define a function to check dataset
    def check_if_path_exists(self):
        return False if os.path.exists(self.path_dataset) else True
        # If the directory exists, return False <-> Otherwise return True to download a new dataset





# Run the codes
if __name__ == '__main__':
    # Download a dataset
    #dataset = CustomDataset('./datasets/0731-VOC Segmentation', year='2012', image_set='train', transforms=None)
    dataset = CustomDataset('./datasets/0731-VOC Segmentation/train')
    pass