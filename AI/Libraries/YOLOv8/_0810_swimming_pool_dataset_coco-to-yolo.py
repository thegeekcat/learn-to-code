# Import modules
import os
import json
import shutil


# Set a path for image directories
image_path = 'c:/datasets/0810-Swimming Dataset coco/train/'

# Set a path for an annotation file in coco format
coco_annotation_path = 'c:/datasets/0810-Swimming Dataset coco/train/_annotations.coco.json'


# Create a folder for a converted YOLO format
os.makedirs('../../outcomes/0810-Swimming Dataset_yolo/train/images/', exist_ok=True)
os.makedirs('../../outcomes/0810-Swimming Dataset_yolo/train/labels/', exist_ok=True)
os.makedirs('../../outcomes/0810-Swimming Dataset_yolo/valid/images/', exist_ok=True)
os.makedirs('../../outcomes/0810-Swimming Dataset_yolo/valid/labels/', exist_ok=True)

# Set a path to save image files in yolo format
yolo_image_copy_path = '../../outcomes/0810-Swimming Dataset_yolo/train/images/'
#yolo_image_copy_path = '../../outcomes/0810-Swimming Dataset_yolo/valid/images/'

# Set a path to save annotation files in yolo format
yolo_annotation_save_path = '../../outcomes/0810-Swimming Dataset_yolo/train/labels/'
#yolo_annotation_save_path = '../../outcomes/0810-Swimming Dataset_yolo/valid/labels/'




# Load COCO annotations
with open(coco_annotation_path, 'r', encoding='utf-8') as f:
    coco_annotation_info = json.load(f)

image_infos = coco_annotation_info['images']
anno_infos = coco_annotation_info['annotations']


for image_info in image_infos:
    image_file_name = image_info['file_name']
    file_name = image_file_name.replace('jpg', '')
    id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']

    for ann_info in anno_infos:
        if ann_info['image_id'] == id:

            category_id = ann_info['category_id']

            """
            Not neccessary lables: 0, 1, 7
            Labels to be changed: 2, 3, 4, 5, 6 -> swimming 0
            """

            if category_id not in [0, 1, 7]:
                x, y, w, h = anno_info['bbox']

                # xywh -> center x, center y, w, h
                x_center = ((x + w) / 2) / image_width
                y_center = ((y + h) / 2) / image_height
                w /= image_width
                h /= image_height

                # Copy images to destination folder (yolo)
                source_image_path = os.path.join(image_path, image_file_name)
                destination_image_path = os.path.join(yolo_image_copy_path, image_file_name)
                shutil.copy(source_image_path, destination_image_path)

                # Write text to file
                yolo_line = f'0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n'
                text_path = os.path.join(yolo_annotation_save_path, f'{file_name}.txt')
                with open(text_path, 'a') as f:
                    f.write(yolo_line)













