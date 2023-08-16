# Import modules
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

# Define a class
class MyDataset(Dataset):
    # Initialize the class
    def __init__(self, dir_data, transforms=None):
        self.label_dictionary = {'Normal': 0, 'Pneumonia_bacteria': 1, 'Pneumonia_virus': 2}
        self.dir_data = glob.glob(os.path.join(dir_data, '*', '*.jpeg'))
        #print(self.dir_data)
        #exit()
        self.transforms = transforms

    # Define __getitem__
    def __getitem__(self, item):
        # Set path
        path = self.dir_data[item]
        #print(path)
        #exit()

        # Get images
        image = Image.open(self.dir_data[item])
        image = image.convert('RGB')
        #print(image)

        # Get labels
        dir_name = path.split('\\')[1]
        #print(dir_name)  # Result: Normal
        label = int(self.label_dictionary[dir_name])
        # print(label)  # Result: 0



        # Set transforms
        if self.transforms is not None:
            image = self.transforms(image)

        #print(image, label)  # Result: <PIL.Image.Image image mode=RGB size=1620x1356 at 0x1DD08AD50D0> 0
        return image, label


    # Define __len__
    def __len__(self):
        return len(self.dir_data)


# Run codes
test = MyDataset('./outcomes/0718-Pneumonia Dataset_splited/Train', transforms=None)
for i in test:
    pass