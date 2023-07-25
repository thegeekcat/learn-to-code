# Import modules
import os
import glob
import shutil
import random
from sklearn.model_selection import train_test_split

# Define a class for the file organizer
class FileManagement:
    # Initalize the class,
    def __init__(self, path_folder_root_original_dataset, path_folder_root_split_dataset):
        self.path_folder_root_original_dataset = path_folder_root_original_dataset
        self.path_folder_root_split_dataset = path_folder_root_split_dataset
        #print('Path of Original Dataset: ', self.path_folder_root_original_dataset)  # Result: 'Path of Original Dataset:  ./datasets/0713-Biscuit Wrappers Dataset'
        #print('Path of Split Dataset: ', self.path_folder_root_split_dataset)  # Result: 'Path of Split Dataset:  ./outcomes/0713-Biscuit Wrappers Dataset_split'

        # Set paths
        self.path_folder_train = os.path.join(self.path_folder_root_split_dataset, 'train')
        self.path_folder_validation = os.path.join(self.path_folder_root_split_dataset, 'validation')
        # print(path_folder_train)       # Result: './outcomes/0713-Biscuit Wrappers Dataset_split\train'
        # print(path_folder_validation)  # Result: './outcomes/0713-Biscuit Wrappers Dataset_split\validation'

        # Create folders
        os.makedirs(self.path_folder_train, exist_ok=True)
        os.makedirs(self.path_folder_validation, exist_ok=True)


    # Define a function to get labels by folder name
    def get_labels_from_folder_names(self):
        folder_names = os.listdir(self.path_folder_root_original_dataset)
        labels = {}
        for i, folder_name in enumerate(folder_names):
            labels[folder_name] = i
        #print(labels.keys())               # Result:
                                            #    dict_keys(['Americana Coconut Cookies', 'Amul Chocolate Cookies', 'Amul Elaichi Rusk', 'Bisk Farm Sugar Free Biscuits', 'Bonn Jeera Bite Biscuits',
                                            #    'Britannia 50-50 Maska Chaska', 'Britannia 50-50 Potazos - Masti Masala', 'Britannia 50-50 Timepass Classic Salted Biscuit', 'Britannia Biscafe Coffee Cracker', 'Britannia Bourbon',
                                            #    'Britannia Chocolush - Pure Magic', 'Britannia Good Day - Chocochip Cookies', 'Britannia Good Day Cashew Almond Cookies', 'Britannia Good Day Harmony Biscuit', 'Britannia Good Day Pista Badam Cookies',
                                            #    'Britannia Little Hearts', 'Britannia Marie Gold Biscuit', 'Britannia Nice Time - Coconut Biscuits', 'Britannia Nutri Choice Sugar Free Cream Cracker Biscuits', 'Britannia Nutrichoice Herbs Biscuits',
                                            #    'Britannia Tiger Glucose Biscuit', 'Britannia Tiger Kreemz - Chocolate Cream Biscuits', 'Britannia Tiger Kreemz - Elaichi Cream Biscuits', 'Britannia Tiger Kreemz - Orange Cream Biscuits', 'Britannia Tiger Krunch Chocochips Biscuit',
                                            #    'Britannia Treat Chocolate Cream Biscuits', 'Britannia Treat Crazy Pineapple Cream Biscuit', 'Britannia Treat Osom Orange Cream Biscuit', 'Britannia Vita Marie Gold Biscuits', 'Cadbury Oreo Chocolate Flavour Biscuit Cream Sandwich',
                                            #    'Canberra Big Orange Cream Biscuits', 'CookieMan Hand Pound Chocolate Cookies', 'Cremica Coconut Cookies', 'Cremica Jeera Lite', 'MARIO Coconut Crunchy Biscuits',
                                            #    'McVities Marie Biscuit', 'Parle 20-20 Cashew Cookies', 'Parle 20-20 Nice Biscuits', 'Parle Happy Happy Choco-Chip Cookies', 'Parle Hide and Seek',
                                            #    'Parle Hide and Seek - Black Bourbon Choco', 'Parle Hide and Seek Caffe Mocha Cookies', 'Parle Hide and Seek Chocolate and Almonds', 'Parle Krackjack Biscuits', 'Parle Magix Sandwich Biscuits - Chocolate',
                                            #    'Parle Milk Shakti Biscuits', 'Parle Monaco Biscuit - Classic Regular', 'Parle Monaco Piri Piri', 'Parle-G Gold Gluco Biscuits', 'Parle-G Original Gluco Biscuits',-
                                            #    'Patanjali Doodh Biscuit', 'Priyagold Butter Delite Biscuits', 'Sagar Coconut Munch Biscuits', 'Sunfeast All Rounder - Cream and Herb', 'Sunfeast Bounce Creme Biscuits',
                                            #    'Sunfeast Dark Fantasy - Choco Creme', 'Sunfeast Dark Fantasy Bourbon Biscuits', 'Sunfeast Dark Fantasy Choco Fills', 'Sunfeast Glucose Biscuits', 'Sunfeast Moms Magic - Fruit and Milk Cookies',
                                            #    'Sunfeast Moms Magic - Rich Butter Cookies', 'Sunfeast Moms Magic - Rich Cashew and Almond Cookies', 'UNIBIC Choco Chip Cookies', 'UNIBIC Pista Badam Cookies', 'UNIBIC Snappers Potato Crackers'])
        #print('Total number of labels: ', len(labels))   # Result: Total number of labels:  65


    # Define a function to create subfolders and copy images
    def make_subfolders_and_split_images(self):
        # Get a list of folder names
        folder_names = os.listdir(self.path_folder_root_original_dataset)
        #print(folder_names)  # Result: ['Americana Coconut Cookies', 'Amul Chocolate Cookies', 'Amul Elaichi Rusk', .... ]

        # Split images as Train and Validation datasets
        for folder_name in folder_names:
            # Get paths including subfolders
            path_folder_original_dataset_with_subfolders = os.path.join(self.path_folder_root_original_dataset, folder_name)
            #print(path_folder_original_dataset_with_subfolders)  # Result: './datasets/0713-Biscuit Wrappers Dataset\Americana Coconut Cookies'

            # Get name of images
            images = os.listdir(path_folder_original_dataset_with_subfolders)
            #print(images)  # Result: '['Americana Coconut Cookies (1).jpg', 'Americana Coconut Cookies (10).jpg', ... ]'

            # Randomly shuffle images
            random.shuffle(images)

            # Creaet subfolders by labels
            path_folder_train_with_subfolders = os.path.join(self.path_folder_train, folder_name)
            path_folder_validation_with_subfolders = os.path.join(self.path_folder_validation, folder_name)
            #print(path_folder_train_with_subfolders)      # Result: './outcomes/0713-Biscuit Wrappers Dataset_split\train\Americana Coconut Cookies'
            #print(path_folder_validation_with_subfolders) # Result: './outcomes/0713-Biscuit Wrappers Dataset_split\validation\Americana Coconut Cookies'
            os.makedirs(path_folder_train_with_subfolders, exist_ok=True)
            os.makedirs(path_folder_validation_with_subfolders, exist_ok=True)

            # Set splitting ratio for Train dataset
            split_index = int(len(images) * 0.9)  # '0.9': 90% of images allocated as Training dataset
            #print(split_index)  # Result: 43
                                 #         54 ...

            # Copy image data to Training dataset folder
            for image in images[:split_index]:
                path_source = os.path.join(path_folder_original_dataset_with_subfolders, image)
                path_destination = os.path.join(path_folder_train_with_subfolders, image)
                #print('Path for source: ', path_source) # Result: 'Path for source:  ./datasets/0713-Biscuit Wrappers Dataset\Americana Coconut Cookies\Americana Coconut Cookies (18).jpg'
                #print('Path for destination: ', path_destination) # Result: 'Path for destination:  ./outcomes/0713-Biscuit Wrappers Dataset_split\train\Americana Coconut Cookies\Americana Coconut Cookies (18).jpg'
                shutil.copyfile(path_source, path_destination)

            # Copy image data to Validation dataset folder
            for image in images[split_index:]:
                path_source = os.path.join(path_folder_original_dataset_with_subfolders, image)
                path_destination = os.path.join(path_folder_validation_with_subfolders, image)
                #print('Path for source: ', path_source)  # Result: 'Path for source:  ./datasets/0713-Biscuit Wrappers Dataset\Americana Coconut Cookies\Americana Coconut Cookies (46).jpg'
                #print('Path for destination: ', path_destination)  # Result: 'Path for destination:  ./outcomes/0713-Biscuit Wrappers Dataset_split\validation\Americana Coconut Cookies\Americana Coconut Cookies (46).jpg'
                shutil.copyfile(path_source, path_destination)

        print('All images are successfully copied!')


    def main(self):
        self.get_labels_from_folder_names()
        self.make_subfolders_and_split_images()



# Run codes
if __name__ == '__main__':
    # Set paths
    path_folder_root_original_dataset = './datasets/0713-Biscuit Wrappers Dataset'
    path_folder_root_split_dataset = './outcomes/0713-Biscuit Wrappers Dataset_split'

    test = FileManagement(path_folder_root_original_dataset, path_folder_root_split_dataset)
    test.main()



