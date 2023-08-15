# Import modules
import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

# Set folder paths for datasets
path_folder_train = './outcomes/0724-Chocolate Dataset_splited/Train/'
path_folder_validation = './outcomes/0724-Chocolate Dataset_splited/Validation/'

# Set csv paths to save csv files
paths_file_csv_train = os.path.join(path_folder_train, 'annotations.csv')
path_file_csv_validation = os.path.join(path_folder_validation, 'annotations.csv')

# Make dataframe
df_annotations_train = pd.read_csv(paths_file_csv_train)
df_annotations_validation = pd.read_csv(path_file_csv_validation)


# Define a function to resize images and rescale bounding boxes
def resize_and_rescale_bbox(img, bbox, target_size):
    # Get size of bounding boxes
    img_width, img_height = img.size
    #print(img_width, img_height) # Result: 5152 3864

    # Resize images
    img = img.resize(target_size, Image.LANCZOS)  # 'Image.LANCZOS'
    img_resized_width, img_resized_height = img.size
    #print(img_resized_width, img_resized_height) # Result: 1280 720

    # Calculate scales of bounding boxes
    x, y, width, height = bbox
    x_scale = target_size[0] / img_width
    y_scale = target_size[1] / img_height

    # Calculate rescaled bounding boxes
    x_center = (x + width / 2) * x_scale
    y_center = (y + height / 2) * y_scale
    rescaled_width = width * x_scale
    rescaled_height = height * y_scale

    rescaled_bbox = (x_center, y_center, rescaled_width, rescaled_height)

    return img, rescaled_bbox



# Define a function to convert dataframe to YOLO format
def convert_to_yolo_format(df_annotation, folder_resource, folder_output, target_size):
    # Get annotations
    for idx, row in tqdm(df_annotation.iterrows(), leave=True):
        #print(row)
        #exit()  # Result: filename                                                        IMG_2052.JPG
                #         file_size                                                            6539401
                #         file_attributes                                                           {}
                #         region_count                                                               4
                #         region_id                                                                  0
                #         region_shape_attributes    {"name":"rect","x":957,"y":1601,"width":1970,"...
                #         region_attributes                 {"":{"hello":false},"candy_type":"Crunch"}
                #         Name: 0, dtype: object

        # Get names of images and labels
        image_name = row['filename']
        label = row['region_id']
        #print(f'Name of Image(Labels): {image_name} ({label})') # Result: Name of Image(Labels): IMG_2052.JPG (0)
        #print('Name of image: ', image_name, ', Label: ', label) # Result: Name of image:  IMG_2052.JPG , Label:  0

        # Set image paths
        path_image_before_yolo = os.path.join(folder_resource, image_name)
        #print('Image path before applying YOLO: ', path_image_before_yolo) # Result: img path:  Image path before applying YOLO:  ./outcomes/0724-Chocolate Dataset_splited/Train/IMG_2052.JPG
        path_image_after_yolo = os.path.join(folder_output, 'images', image_name)
        #print('Image path after applying YOLO: ', path_image_after_yolo)  # Result: Image path after applying YOLO:  ./outcomes/0724-Chocolate Dataset_yolo/Train\images\IMG_2052.JPG


        # Get Bounding boxes information
        shape_attributes = json.loads(row['region_shape_attributes'])
        #print('shape_attributes: ', shape_attributes) # Result: shape_attributes:  {'name': 'rect', 'x': 957, 'y': 1601, 'width': 1970, 'height': 1196}

        x = shape_attributes['x']
        y = shape_attributes['y']
        width = shape_attributes['width']
        height = shape_attributes['height']
        print(f'x: {x}, y: {y}, width: {width}, height: {height}')  # REsult: x: 957, y: 1601, width: 1970, height: 1196



        # Read images to get sizes
        img = Image.open(path_image_before_yolo)

        # Resize images and Rescale bounding boxes
        img, rescaled_bbox = resize_and_rescale_bbox(img, (x, y, width, height), target_size)

        # Save images
        img.save(path_image_after_yolo)

        # Calculate information of bounding boxes
        x_center, y_center, width, height = rescaled_bbox
        x_center /= target_size[0]
        y_center /= target_size[1]
        normalized_width = width / target_size[0]
        normalized_height = height / target_size[1]
        #print(f'x_center: {x_center}, y_center: {y_center}, normalized_width: {normalized_width}, normalized_height: {normalized_height} ')
        # Result: x_center: 0.37694099378881984, y_center: 0.5690993788819876, normalized_width: 0.3823757763975155, normalized_height: 0.3095238095238095


        # Get Class IDs
        class_id = label
        #print(class_id)  # Result: 0


        # Create label files
        file_labels = os.path.splitext(image_name)[0] + '.txt'
        print('file_labels: ', file_labels)  # Result: file_labels:  IMG_2052.txt
        path_labels = os.path.join(folder_output, 'labels', file_labels)
        #print('path_labels: ', path_labels)  # Result: path_labels:  ./outcomes/0724-Chocolate Dataset_yolo/Train\labels\IMG_2052.txt


        # Save as text files
        with open(path_labels, 'a') as f:
            line = f'{class_id} {x_center: .6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}\n'
            f.write(line)


'''
## Folder structures
0724-Chocolate Dataset
    Train
        images
            aaa.png
        labels
            aaa.txt
    Validation
        images
            bbb.png
        labels
            bbb.txt
'''

# Create folders
folder_yolo_train = './outcomes/0724-Chocolate Dataset_yolo/Train'
folder_yolo_validation = './outcomes/0724-Chocolate Dataset_yolo/Validation'
os.makedirs(os.path.join(folder_yolo_train, 'images'), exist_ok=True)
os.makedirs(os.path.join(folder_yolo_train, 'labels'), exist_ok=True)
os.makedirs(os.path.join(folder_yolo_validation, 'images'), exist_ok=True)
os.makedirs(os.path.join(folder_yolo_validation, 'labels'), exist_ok=True)


# Set target size
target_size = (1280, 720)


# Run the function
convert_to_yolo_format(df_annotations_train, path_folder_train, folder_yolo_train, target_size)

