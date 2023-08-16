# Import modules
import os
import cv2
import glob
from torch.utils.data import Dataset


# Define a class
class MyUSLicensePlatesDataset(Dataset):

    # Initalize the class
    def __init__(self, dir_data, transform=None):
        # Set a directory path    # ./data/0718-US_license_plates_dataset/Train/*/*.jpg
        self.dir_data = glob.glob(os.path.join(dir_data, '*', '*.jpg'))
        self.transform = transform
        self.label_dict = self.create_label_dict()


    # Define Create Label Dict
    def create_label_dict(self):
        # Initialize dictionary
        label_dict = {}

        # Get labels
        for filepath in self.dir_data:
            label = os.path.basename(os.path.dirname(filepath))
            #print(label)  # Result: ALABAMA

            if label not in label_dict:
                label_dict[label] = len(label_dict)

            #print(label_dict)  # Result: {'ALABAMA': 0}

        return label_dict



    # Define __get__item
    def __getitem__(self, item):
        # Get paths
        path_images = self.dir_data[item]
        #print(path_images)  # Result: ./data/0718-US License Plates Dataset/Train\ALABAMA\001.jpg


        # Load images
        image = cv2.imread(path_images)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image)

        # Get labels
        label = os.path.basename(os.path.dirname(path_images))
        label_idx = self.label_dict[label]
        #print(label, label_idx)


        # Apply augmentation
        if self.transform is not None:
            image = self.transform(image=image)['image']
            #Sprint(image)

        return image, label_idx

    def __len__(self):
        return len(self.dir_data)


# Run codes
test = MyUSLicensePlatesDataset('./data/0718-US License Plates Dataset/Train')
for i in test:
    pass













