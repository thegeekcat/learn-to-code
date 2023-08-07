# Import modules
import cv2
import json
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
from _0804_parking_dataset_train import cfg



checkpoint_file_path = './work_dirs/0804_Parking_dataset/epoch_3.pth'

import torch
device = torch.device("cuda")
model = init_detector(cfg, checkpoint_file_path, device=device)


with open('c:/datasets/0804-Parking_Dataset_coco/test.json', 'r', encoding='utf-8') as f :
    image_infos = json.loads(f.read())

# Confidence Score def => 0.5

if __name__ == '__main__' :

    score_threshold = 0.65
    for img_info in image_infos['images'] :
        file_name = img_info['file_name']
        image_path = os.path.join('./c:/datasets/0804-Parking_Dataset_coco/images' , file_name)
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = inference_detector(model, img)

        for idx, result in enumerate(results) :
            if len(result) == 0 :
                continue
            result_filtered = result[np.where(result[:,4] > score_threshold)]
            for i in range(len(result_filtered)) :
                x1 = int(result_filtered[i, 0])
                y1 = int(result_filtered[i, 1])
                x2 = int(result_filtered[i, 2])
                y2 = int(result_filtered[i, 3])
                print(x1, y1, x2 , y2, float(result_filtered[i, 4]))

                img = cv2.rectangle(img, (x1, y1, x2, y2), (0,255,0), 2)

        cv2.imshow('Test', img)
        cv2.waitKey(0)