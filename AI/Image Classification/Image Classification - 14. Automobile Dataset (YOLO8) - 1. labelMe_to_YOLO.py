# Import modules
import os
import glob
import cv2
import random
from tqdm import tqdm

# Make folders
"""
yolo_dataset
    train
        images
        labels
    validation
        images
        labels
"""
os.makedirs('./yolo_dataset/train/images', exist_ok=True)
os.makedirs('./yolo_dataset/train/labels', exist_ok=True)
os.makedirs('./yolo_dataset/validation/images', exist_ok=True)
os.makedirs('./yolo_dataset/validation/labels', exist_ok=True)


# Get a list of files
list_file_txt = glob.glob('./yolo_dataset/train/*.txt')
print(list_file_txt)
exit()

# Set a validation ratio
validation_ratio = 0.1
validation_size = int(len(txt_file_list) * validation_ratio)

# Get files
files_train = list_file_txt[validation_size:]
files_validation = list_file_txt[:validation_size]
for file in tqdm(files_train):
    file_name = os.path.basename(file)
    print(file_name)

    img = cv2.imread(f'./yolo_dataset/train/{file_name}.png')
    print(img)

    # Save images
    #cv2.imwrite(f'./outcomes/0725-Automobile Dataset_splited_YOLOv8/train/images/{file_name}.png', img)

    # Get Height and Width
    img_height, img_width, _ = img.shape

    # Get lines
    with open(file, 'r', encoding='utf-8') as f:
        # Initialize a list of lines
        list_lines = []

        # Load line info
        lines = f.readlines()
        for line in lines:
            # Get each line
            line = list(map(float, line.strip().split(' ')))

            # Get class names
            class_name = int(len(line[0]))

            # Get axes
            x_min = float(min(line[5], line[7]))
            y_min = float(min(line[6], line[8]))
            x_max = float(max(line[1], line[3]))
            y_max = float(max(line[2], line[4]))

            # Calculate center values
            x_center = float(((x_min + x_max) / 2) / img_width)
            y_center = float(((y_min + y_max) / 2) / img_height)

            # Calculate width and height
            width = abs(x_max - x_min) / img_width
            height = abs(y_max - y_min) / img_height

            # Add to list
            list_lines.append([class_name, x_center, y_center, width, height])

        #
        with open('./outcomes/0725-Automobile Dataset_splited_YOLOv8/train/labels' + file_name + '.txt', 'w') as f:
            for line in list_lines:
                f.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + str[line[4]] + '\n')


