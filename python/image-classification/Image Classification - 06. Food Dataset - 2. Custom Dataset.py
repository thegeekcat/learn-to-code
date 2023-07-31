# Import modules
import os
import cv2
import glob

from torch.utils.data import Dataset

# Define a Dataset class
class MyFoodDataset(Dataset):
    # Initialize the class
    def __init__(self, directory_data, transform=None):
        # directory_data = './data/0717-Food Dataset/Train'

        # Set directory
        self.directory_data = glob.glob(os.path.join(directory_data, '*', '*.jpg'))
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
            # print(label)  # Result: burgrer
            # exit()

        #print(label_dictionary)
        # Result: {'burger': 0, 'butter_naan': 1, 'chai': 2, 'chapati': 3, 'chole_bhature': 4,
        #          'dal_makhani': 5, 'dhokla': 6, 'fried_rice': 7, 'idli': 8, 'jalebi': 9,
        #          'kaathi_rolls': 10, 'kadai_paneer': 11, 'kulfi': 12, 'masala_dosa': 13, 'momos': 14,
        #          'paani_puri': 15, 'pakode': 16, 'pav_bhaji': 17, 'pizza': 18, 'samosa': 19}

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
test = MyFoodDataset('./data/0717-Food Dataset/Train', transform=None)
for i in test:
    pass