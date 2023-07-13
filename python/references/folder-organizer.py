############## Purpose of this class ##############
# < File Organizer >
#  - Copy or Move all files from subdirectories to new folders
###################################################

# Import modules
import os
import shutil
import glob


# Define a class for Moving Images
class FolderOrganizer:
    # Initialize the class
    def __init__(self, org_folder):
        self.org_folder = org_folder

    # Define Move Images part
    def move_images(self):
        # Set file paths
        ## 'org_folder': './data/0713-Sound Images Reference'
        ## File path: './data/0713-Sound Images Reference\MelSepctrogram\blues\blues.00000_augmented_noise.png'
        file_path_list = glob.glob(os.path.join(self.org_folder, '*', '*', '*.png'))
        print(file_path_list)
        exit()


        # Move images to new folders
        for file_path in file_path_list:
            # Check file_path
            #print(file_path)
            folder_name = file_path.split('\\')[1]
            #print(folder_name)
            destination_path = ''

            # Move to folders
            if folder_name == 'MelSpectrogram':
                destination_path = './data/0713-Sound Images/MelSpectrogram/'
                shutil.move(file_path, destination_path)
                print(f'File is transfering to {folder_name}')
            elif folder_name == 'STFT':
                destination_path = './data/0713-Sound Images/STFT/'
                shutil.move(file_path, destination_path)
                print(f'File is transfering to {folder_name}')
            elif folder_name == 'waveshow':
                destination_path = './data/0713-Sound Images/waveshow/'
                shutil.move(file_path, destination_path)
                print(f'File is transfering to {folder_name}')

# Move images to a new folder
FileManagement = FolderOrganizer('./data/0713-Sound Images Reference')
FileManagement.move_images()

