# Import modules
import os
import random
import shutil

##### Set Initial Folders #####
# Set Dataset Folder path
label_folder_path = './data/0714-Cotton Weed ID15 Reference'

# Set New Folder path for splitting
dataset_folder_path = './outcomes/0714-Cotton Weed ID15'

# Set folder paths for Train and Validation
train_folder_path = os.path.join(dataset_folder_path, 'Train')
val_folder_path = os.path.join(dataset_folder_path, 'Validation')

# Create folders
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)


#  Labels: ['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory',
#           'Nutsedge', 'PalmerAmaranth', 'Prickly Sida', 'Purslane', 'Ragweed',
#           'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress', 'Waterhemp']

# Get folder names
org_folders = os.listdir(label_folder_path)
#print(org_folders)

for org_folder in org_folders:
    # Get full paths
    org_folder_full_path = os.path.join(label_folder_path, org_folder)
    #print(org_folder_full_path)   # Result: ./data/0714-Cotton Weed ID15 Reference\Carpetweeds

    # Get name of images
    images = os.listdir(org_folder_full_path)
    #print(images)
    random.shuffle(images)  # Randomly shuffle images

    # Create folders for labels
    train_label_folder_path = os.path.join(train_folder_path, org_folder)
    val_label_folder_path = os.path.join(val_folder_path, org_folder)
    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)

    # Set percentage of Training Dataset
    split_index = int(len(images) * 0.9)  # 90% of images set as Training Dataset

    # Copy image files to Training Dataset Folder
    for image in images[:split_index]:
        source_path = os.path.join(org_folder_full_path, image)
        destination_path = os.path.join(train_label_folder_path, image)  # ./outcomes/0714-Cotton Weed ID 15/train
        shutil.copyfile(source_path, destination_path)

    # Copy image files to Validation Dataset Folder
    for image in images[split_index:]:
        source_path = os.path.join(org_folder_full_path, image)
        destination_path = os.path.join(val_label_folder_path, image)  # ./outcomes/0714-Cotton Weed ID 15/val
        shutil.copyfile(source_path, destination_path)

    print('Files copied successfully!')





