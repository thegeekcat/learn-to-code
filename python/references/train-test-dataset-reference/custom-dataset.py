# Import modules
import glob
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True



# Define a class
class CustomDataset_CottonWeed(Dataset):

    # Initialize teh class
    def __init__(self, data_dir, transforms=None):
        # Set Data Directory
        #print('data_dir: ', data_dir)
        # 'data_dir': './outcomes/0714-Cotton Weed ID15/Train/'
        self.data_dir = glob.glob(os.path.join(data_dir, '*', '*.jpg'))
        #print('self.data_dir: ', self.data_dir)

        # Set transforms
        self.transforms = transforms
        self.label_dict = {'Carpetweeds':0, 'Crabgrass':1,'Eclipta' : 2, 'Goosegrass':3 ,
                           'Morningglory':4, 'Nutsedge':5, 'PalmerAmaranth':6, 'Prickly Sida':7,
                           'Purslane':8, 'Ragweed':9, 'Sicklepod':10, 'SpottedSpurge':11,
                           'SpurredAnoda':12, 'Swinecress':13, 'Waterhemp':14}
        #print('Label Dictionary: ', self.label_dict)

    # Set Get Item
    def __getitem__(self, item):
        # Get Image Paths
        image_path = self.data_dir[item]
        #print('image_path: ', image_path)

        # Get Label Names
        label_name = image_path.split('\\')[1]
        label = self.label_dict[label_name]
        #print('Label: ', label)

        # Get Images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, image_path

    # Get Length
    def __len__(self):
        return len(self.data_dir)

# Run the code
# path_train_data = './outcomes/0714-Cotton Weed ID15/Train/'
# test = CustomDataset_CottonWeed(path_train_data, transforms = None)
# for i in test:
#     print(i)



