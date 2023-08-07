# Import modules
import numpy as np
import cv2
import json
import os
import torch
from mmdet.apis import init_detector, inference_detector
from _0807_website_dataset_config import cfg

# Set a path of checkpoint file
checkpoint_file_path = './work_dirs/0807-Website-screenshot-dataset/_0807_website_dataset_epoch_16.pth'


# Set device
device = torch.device('cuda')

# Set model
model = init_detector(cfg, checkpoint_file_path, device=device)

# Load json file
with open('c:/datasets/0807-Website Screenshots Dataset_coco/test/_annotations.coco.json',
          'r',
          encoding='utf-8') as f:
    image_infos = json.loads(f.read())

# Run codes
if __name__ == '__main__':

    # Set a threshold
    score_treshold = 0.65

    # Get results
    for img_info in image_infos['images']:
        file_name = img_info['file_name']
        image_path = os.path.join('c:/datasets/0807-Website Screenshots Dataset_coco/test/', file_name)
        img = cv2.imread(image_path)

        results = inference_detector(model, img)

        # Display rectangles
        for idx, result in enumerate(results):
            if len(result) == 0:
                continue
            results_filtered = result[np.where(result[:, 4] > score_treshold)]
            for i in range(len(results_filtered)):
                x1 = int(results_filtered[i, 0])
                y1 = int(results_filtered[i, 1])
                x2 = int(results_filtered[i, 2])
                y2 = int(results_filtered[i, 3])
                print(x1, y1, x2, y2, float(results_filtered[i, 4]))

                img = cv2.rectangle(img, (x1, y1, x2, y2), (0,255,0), 2)

            cv2.imshow('Test', img)
            cv2.waitKey(0)
