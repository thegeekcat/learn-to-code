###### Task Lisks #####
# 1. Load a json file to get axises to crop images
# 2. Crop images based on axises
# 3. Add padding to resize image as 256x256
# 4. Save files in train & val (train:val = 9:1)
########################


# Import modules
import numpy as np
import json
import os
import glob
import torch
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


# Set a list of labels
labels = ['crease', 'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole',
          'rolled_pit', 'silk_spot', 'waist_folding', 'water_spot', 'welding_line']

# Define modifying images
def crop_and_save_image(path_json_file, dir_output, train_ratio=0.9):
    # Load a json file
    with open(path_json_file, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    #print(data_json)
    #exit()


    # Generate folders
    #  Result: ./outcomes/0717-Damaged Metal Dataset/Train
    dir_train = os.path.join(dir_output, 'Train')
    dir_validation = os.path.join(dir_output, 'Validation')
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_validation, exist_ok=True)

    # Generate subfolders for labels
    for label in labels:
        dir_labels_train = os.path.join(dir_train, label)
        dir_labels_validation = os.path.join(dir_validation, label)
        os.makedirs(dir_labels_train, exist_ok=True)
        os.makedirs(dir_labels_validation, exist_ok=True)

    # Get information
    for information in tqdm(data_json.keys()):   # Get 'keys'
        # print(type(information))
        # exit()
        json_image = data_json[information]
        # print(json_image)  # Result: {'information': 'img_01_3402617700_00001.jpg',
        #                    #          'width': 2048,
        #                    #          'height': 1000,
        #                    #          'anno': [{'label': 'crescent_gap', 'bbox': [1738, 806, 1948, 993]}]}
        # exit()
        filename = json_image['filename']
        width = json_image['width']
        height = json_image['height']
        bboxes = json_image['anno']
        #print(file_name, ',', width, 'x',height)

        # Load images
        path_image = os.path.join('./data/0717-Damaged Metal Dataset/Images', filename)
        image = Image.open(path_image)
        image = image.convert('RGB')
        #print(path_image)


        # Crop images
        for bbox_idx, bbox in enumerate(bboxes):  # Reason of using 'bbox_idx': Some images have more than one bounding boxes
            label_name = bbox['label']
            bbox_xyxy = bbox['bbox']
            x1, y1, x2, y2 = bbox_xyxy
            #print(label_name, bbox_xyxy)

            # Crop images
            image_cropped = image.crop((x1, y1, x2, y2))

            # Add Padding
            width_, height_ = image_cropped.size
            if width_ > height_:
                image_padded = Image.new(image_cropped.mode, (width_, width_), (0,))
                padding = (int(width_ - height_) / 2)
            else:
                image_padded = Image.new(image_cropped.mode, (height_, height_), (0,))
                padding = (int(height_ - height_) / 2, 0)


            # Resize images
            size = (255, 255)
            resize_image = F.resize(image_cropped, size)

            # Save images to each folder
            if np.random.rand() < train_ratio:
                dir_save = os.path.join(dir_train, label_name)
            else:
                dir_save = os.path.join(dir_validation, label_name)


            ### REFERENCES ###
            # for label_name in label_names:
            #     label_dir = os.path.join(root_dir, label_name)
            #     files = os.listdir(label_dir)
            #     np.random.shuffle(files)  # Shuffle the files randomly
 
            # for i, file in enumerate(files):
            #     if i < len(files) * train_ratio:
            #         save_dir = os.path.join(train_dir, label_name)
            #     else:
            #         save_dir = os.path.join(val_dir, label_name)
                

            path_save = os.path.join(dir_save, f'{filename}_{label_name}_{bbox_idx}.png') # Result: ./outcomes/0717-Damaged Metal Dataset\Train\crescent_gap\img_01_3402617700_00001.jpg_crescent_gap_0.png
            # print(path_save)
            # exit()
            image_padded.save(path_save)

# Run codes
if __name__ == '__main__':
    path_json_file = './data/0717-Damaged Metal Dataset/Annotation/annotation.json'
    dir_output = './outcomes/0717-Damaged Metal Dataset_cropped'

    crop_and_save_image(path_json_file, dir_output)