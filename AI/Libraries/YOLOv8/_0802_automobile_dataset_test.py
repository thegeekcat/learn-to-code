# Import modules
import os
import glob
import cv2
from ultralytics import YOLO

print(os.getcwd())
#exit()

# Set model
model = YOLO('./outcomes/outcomes-0802-automobile/0802_best/weights/best.pt')
path_data = './datasets/data-0802-automobile/test/'

list_path_data = glob.glob(os.path.join(path_data, '*.png'))
# print(list_path_data)
# exit()

#path_temp='D:/OneDrive/AISchoolVM/models/0802-YOLOv8/datasets/data-0802-automobile/test/syn_00007.png'
# path_temp='./test/syn_00007.png'
# results = model.predict(path_temp, save=False, imgsz=640, conf=0.7)
# exit()

# Get detailed info
for path in list_path_data:
    # Read images
    image = cv2.imread(path)
    #print('path: ', path)
    # exit()

    # Get names
    names = model.names
    #print(names)
    #exit()



    # Get results
    results = model.predict(path, save=False, imgsz=640, conf=0.7)
    print(results)
    exit()
    boxes = results[0].boxes
    results_info = boxes

    # Get number of classes
    cls_numbers = results_info.cls

    # Score of bounding boxes
    conf_numbers = results_info.conf

    # Get axes of bounding boxes
    box_xyxy = results_info.xyxy

    for bbox, cls_idx, conf_idx in zip(box_xyxy, cls_numbers, conf_numbers):
        print(bbox, cls_idx, conf_idx)

    print(results)
    exit()
