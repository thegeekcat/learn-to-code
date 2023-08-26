# Import modules
import os
import random
import shutil


# Set folder paths for original data
path_folder_original_data = './data/0718-Pneumonia Dataset'

# Set folder path for organized files
path_folder_splited_data = './outcomes/0718-Pneumonia Dataset_splited'

# Set folder paths for Train and Validation
path_folder_train = os.path.join(path_folder_splited_data, 'Train')
path_folder_validation = os.path.join(path_folder_splited_data, 'Validation')
#print(path_folder_train)  # Result: ./outcomes/0718-Pneumonia Dataset_splited\Train
#print(path_folder_validation)   # Result: ./outcomes/0718-Pneumonia Dataset_splited\Validation

# Create folders
os.makedirs(path_folder_train, exist_ok=True)
os.makedirs(path_folder_validation, exist_ok=True)


# Get subfolders
name_subfolders = os.listdir(path_folder_original_data)
#print(name_subfolders)  # Result: ['Normal', 'Pneumonia_bacteria', 'Pneumonia_virus']

# Create subfolders
for name_subfolder in name_subfolders:
    path_folder_full_original_data = os.path.join(path_folder_original_data, name_subfolder)
    #print(path_folder_full_original_data)  # Result: ./data/0718-Pneumonia Dataset\Normal

    # Get names of images
    images = os.listdir(path_folder_full_original_data)
    #print(images)  # Result: 'IM-0115-0001.jpeg', 'IM-0117-0001.jpeg', 'IM-0119-0001.jpeg',

    # Shuffle randoms
    random.shuffle(images)

    # Create folders by label
    path_folder_label_train = os.path.join(path_folder_train, name_subfolder)
    #print(path_folder_label_train)  # Result: ./outcomes/0718-Pneumonia Dataset_splited\Train\Normal
                                     #         ./outcomes/0718-Pneumonia Dataset_splited\Train\Pneumonia_bacteria
                                     #         ./outcomes/0718-Pneumonia Dataset_splited\Train\Pneumonia_virus
    path_folder_label_validation = os.path.join(path_folder_validation, name_subfolder)
    #print(path_folder_label_validation)  # Result: ./outcomes/0718-Pneumonia Dataset_splited\Validation\Normal
                                          #         ./outcomes/0718-Pneumonia Dataset_splited\Validation\Pneumonia_bacteria
                                          #         ./outcomes/0718-Pneumonia Dataset_splited\Validation\Pneumonia_virus
    os.makedirs(path_folder_label_train, exist_ok=True)
    os.makedirs(path_folder_label_validation, exist_ok=True)

    # Set split ratio
    split_index = int(len(images) * 0.9)

    # Split 90% of images to Train folder
    for image in images[:split_index]:
        path_source = os.path.join(path_folder_full_original_data, image)
        path_destination = os.path.join(path_folder_label_train, image)
        shutil.copyfile(path_source, path_destination)

    # Split 10% of images to Validation folder
    for image in images[split_index:]:
        path_source = os.path.join(path_folder_full_original_data, image)
        path_destination = os.path.join(path_folder_label_validation, image)
        shutil.copyfile(path_source, path_destination)

    print('Files copied successfully!')
