# Import modules
import cv2
import json
import numpy as np
import os
import torch
from mmdet.apis import init_detector, inference_detector
from _0808_flowchart_dataset_config import cfg

# Set a checkpoint file path
checkpoint_file_path = './work_dirs/0808_Flowchart_dataset/r50_1x_AdamW_lr-0.005/latest.pth'


# Set model
device = torch.device('cuda')
model = init_detector(cfg, checkpoint_file_path, device=device)

# Load json file
with open('c:/datasets/0808-Flow chart detection_coco/test/_annotations.coco.json', 'r', encoding='utf-8') as f:
    image_infos = json.loads(f.read())

# Run codes
if __name__ == '__main__':
    # Set threshold
    score_threshold = 0.65

    # Display images
    for img_info in image_infos['images']:
        # Get image files
        file_name = img_info['file_name']
        image_path = os.path.join('c:/datasets/0808-Flow chart detection_coco/test/', file_name)
        img = cv2.imread(image_path)

        # Get results
        results = inference_detector(model, img)
        for idx, result in enumerate(results):
            if len(result) == 0:
                continue
            result_filtered = result[np.where(result[:, 4] > score_threshold)]
            for i in range(len(result_filtered)):
                x1 = int(result_filtered[i, 0])
                y1 = int(result_filtered[i, 1])
                x2 = int(result_filtered[i, 2])
                y2 = int(result_filtered[i, 3])
                print(x1, y1, x2, y2, float(result_filtered[i, 4]))

                img = cv2.rectangle(img, (x1, y1, x2, y2), (0, 255, 0), 2)

        cv2.imshow('Test', img)
        cv2.waitKey(0)





