# Import modules
import os
import cv2
import glob
from torch.utils.data import Dataset

# Define a Dataset class
class MyDataset(Dataset):
    # Initialize the class
    def __init__(self, directory_data, transform=None):
        # directory_data = './outcomes/0717-Damaged Metal Dataset_cropped/Train'

        # Set directory
        self.directory_data = glob.glob(os.path.join(directory_data, '*', '*.png'))
        self.transform = transform
        self.label_dictionary = self.create_label_dict()

    # Define Create Label Dictionary
    def create_label_dict(self):
        # Initialize dictionary
        label_dictionary = {}

        # Set Labels
        for filepath in self.directory_data:
            label = os.path.basename(os.path.dirname(filepath))

            if label not in label_dictionary:
                label_dictionary[label] = len(label_dictionary)
            #print(label)  # Result: crease
            #exit()

        #print(label_dictionary)




        return label_dictionary
    #
    def __getitem__(self, item):
        # Set File Paths
        image_filepath = self.directory_data[item]

        # Load Images
        img = cv2.imread(image_filepath)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get Labels
        label = os.path.basename(os.path.dirname(image_filepath))
        #print(label)  # Result: kulfi
        label_idx = self.label_dictionary[label]
        # print(label, label_idx)  # Result: burger 0
        # exit()

        # Call methods for augmentations
        if self.transform is not None:
            image = self.transform(image = img)['image']
            #print(image)
            #exit()

        # print(image)
        # exit()
        return image, label_idx


    def __len__(self):
        return len(self.directory_data)


# Run codes
test = MyDataset('./outcomes/0717-Damaged Metal Dataset_cropped/Train', transform=None)
for i in test:
    pass