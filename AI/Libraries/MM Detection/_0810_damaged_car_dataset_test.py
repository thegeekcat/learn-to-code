# Import modules
import numpy as np
import os
import cv2
import glob

from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.apis import set_random_seed
from mmdet.apis import init_detector, inference_detector

@DATASETS.register_module(force=True)

# Define a class
class DamagedCarDataset(CocoDataset):
    CLASSES = ('damage',)

# Set a config file
config_file = './configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)
#print(cfg.pretty_text)

# Set a dataset default setting
cfg.dataset_type = 'DamagedCarDataset'
cfg.data_root = 'c:/datasets/0810-Damaged Car Dataset/'

# Set a default setting for train/val/test directories
cfg.data.train.type = 'DamagedCarDataset'
cfg.data.train.ann_file = 'c:/datasets/0810-Damaged Car Dataset/train/COCO_train_annos.json'
cfg.data.train.img_prefix = 'c:/datasets/0810-Damaged Car Dataset/train/'
cfg.data.train.pipeline[2].img_sclae=(500, 500)
cfg.data.train.pipeline[3].flip_ratio=0.3

cfg.data.val.type = 'DamagedCarDataset'
cfg.data.val.ann_file = 'c:/datasets/0810-Damaged Car Dataset/val/COCO_val_annos.json'
cfg.data.val.img_prefix = 'c:/datasets/0810-Damaged Car Dataset/val/'
cfg.data.val.pipeline[1].img_scale=(500, 500)

cfg.data.test.type = 'DamagedCarDataset'
cfg.data.test.ann_file = 'c:/datasets/0810-Damaged Car Dataset/val/COCO_val_annos.json' # No json file for test data, so use val json file
cfg.data.test.img_prefix = 'c:/datasets/0810-Damaged Car Dataset/val/'
cfg.data.test.pipeline[1].img_scale=(500, 500)
#print(cfg.pretty_text)


# Set class numbers
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1
cfg.model.rpn_head.anchor_generator.scales = [4]

# Set device and random seed
cfg.device = 'cuda'
set_random_seed(85, deterministic=False)


# Run codes
if __name__ == '__main__':
    # Set a path
    image_path = 'c:/datasets/0810-Damaged Car Dataset/test'
    image_path_list = glob.glob(os.path.join(image_path, '*.jpg'))
    checkpoint_file_path = './work_dirs/0810_Damaged_car_dataset/mask_rcnn_r50_fpn_1x/epoch_75_from_lecturer.pth'

    # Build a model
    model = build_detector(cfg.model, train_cfg=cfg.get('train.cfg'), test_cfg=cfg.get('test.cfg'))

    # Load checkpoints
    model = init_detector(cfg, checkpoint_file_path, device=cfg.device)

    # Change a mode in evaluation
    model.eval()

    # Display images
    for path in image_path_list:
        img = cv2.imread(path)
        #cv2.imshow('Test Images', img)
        #cv2.waitKey(0)

        # Inference
        result = inference_detector(model, img)
        bbox_result, segm_result = result
        #print(bbox_result) # Result: [array([[4.43200500e+02, 2.18345871e+02, 7.22152100e+02, 2.85192108e+02, 8.74514043e-01], [4.97272461e+02, 6.54953857e+02, 5.84616760e+02, 7.21037476e+02, 6.08999372e-01],
        #print(segm_result) # Result: [[array([[False, False, False, ..., False, False, False], [False, False, False, ..., False, False, False], [False, False, False, ..., False, False, False],

        # Copy a layer for overlay
        overlay = img.copy()

        # Get details
        for bbox, segm in zip(bbox_result[0], segm_result[0]):
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Display damaged parts
        if score > 0.35:
            # Get mask areas
            mask = segm
            #print(mask)  # Result: [[False False False ... False False False]
                          #         [False False False ... False False False]


            # Get mask areas in a binary type
            binary_mask = (mask > 0).astype(np.uint8) * 255   # '(mask>0)': 1 = True, 0 = False
            #print(binary_mask) # Result: [[0 0 0 ... 0 0 0]
                                #          [0 0 0 ... 0 0 0]

            # Set RGB channels for overlay layers
            overlay[binary_mask > 0, 0] = 0 # Blue channel
            overlay[binary_mask > 0, 1] = 255 # Green channel
            overlay[binary_mask > 0, 2] = 0   # Red channel

            # Draw a rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display images
            cv2.imshow('Test', overlay)
            cv2.waitKey(0)