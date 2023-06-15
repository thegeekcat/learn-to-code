# Import modules
import json
import os
import cv2
import glob
import numpy as np



# Set file path
json_dir = './data/0615anno/'
json_paths = glob.glob(os.path.join(json_dir, '*json'))

# Set a label
label_dict = {'수각류': 0}

# Get info of JSON files
for json_path in json_paths:

    # Read json files
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Get variables
    images_info = json_data['images']
    annotations_info = json_data['annotations']
    

    # Get info from images
    filename = images_info['filename']
    image_id = images_info['id']
    image_width = images_info['width']
    image_height = images_info['height']

    #  Resize images -> 4:3
    new_width = 1024
    new_height = 768

    # Get info of annotations
    for ann_info in annotations_info:

        if image_id == ann_info['image_id']:

            # Read images
            image_path = os.path.join('./data/0615images/', filename)
            image = cv2.imread(image_path)

            # Set scales of images
            scale_x = new_width / image.shape[1]  # X-axis scale
            scale_y = new_height / image.shape[0] # Y-axis scale

            # Resize images
            image_resized = cv2.resize(image, (new_width, new_height))


            # Set variables
            category_name = ann_info['category_name']
            polygons = ann_info['polygon']

            # Generate coordinates of polygons
            points = []
            for polygon_info in polygons:
                x = polygon_info['x']
                y = polygon_info['y']

                x_resized = int(x * scale_x)
                y_resized = int(y * scale_y)

                points.append((x_resized, y_resized))

            # Draw polygons
            cv2.polylines(image_resized,
                          [np.array(points, np.int32).reshape((-1, 1, 2))],
                          isClosed = True,
                          color = (0, 255, 0),
                          thickness = 2)
            
            # Calculate coordinates for bounding boxes
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            x_min = min(x_coordinates)
            y_min = min(y_coordinates)
            x_max = max(x_coordinates)
            y_max = max(y_coordinates)

            # Draw bounding boxes
            cv2.rectangle(image_resized,
                          (x_min, y_min),
                          (x_max, y_max),
                          (0, 0, 255),
                          2)
            
            # Calculate coordinates in YOLO format
            x_center = ((x_max + x_min) / (2 * new_width))
            y_center = ((y_max + y_min) / (2 * new_height))
            yolo_w = (x_max - x_min) / new_width
            yolo_y = (y_max - y_min) / new_height

            # Set filenames to remove extension parts
            image_name_temp = filename.replace('.jpg', '')

            # Get label numbers
            label_number = label_dict[category_name]

    # Save as a text file
    path_annotations = './data/0615anno/'
    os.makedirs(path_annotations, exist_ok=True)
    with open(f'{path_annotations}/{image_name_temp}.txt', 'a') as f:
        f.write(f'{label_number}, {x_center}, {y_center}, {yolo_w}, {yolo_y} \n')

    # Visualization
    cv2.imshow('Polygon', image_resized)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()




