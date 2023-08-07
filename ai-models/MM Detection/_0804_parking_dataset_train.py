# Import modules
import cv2
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes
from mmcv import Config

# Set decorators
@DATASETS.register_module(force=True)

# Define a class for dataset
class ParkingDataset(CocoDataset):
    CLASSES = ('세단(승용차)', 'SUV', '승합차', '버스','학원차량(통학버스)',
               '트럭','택시','성인','어린이','오토바이',
               '전동킥보드','자전거','유모차','쇼핑카트')

# Set config file to use 'Dynamic RCNN Model'
config_file = './configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)
#print(cfg)  # 'optimizer': { ... 'lr': 0.02, ... }`


### Optimize config parameters
# Set Learning Rate
cfg.optimizer.lr = 0.0025

# Set Dataset Path
cfg.dataset_type = 'ParkingDataset'
cfg.data_root = 'c:/datasets/0804-Parking_Dataset_coco/'
#print(cfg)

# Set parameters for Train dataset
cfg.data.train.type = 'ParkingDataset'
cfg.data.train.ann_file = 'c:/datasets/0804-Parking_Dataset_coco/train.json'
cfg.data.train.img_prefix = 'c:/datasets/0804-Parking_Dataset_coco/images/'

# Set parameters for Validation dataset
cfg.data.val.type = 'ParkingDataset'
cfg.data.val.ann_file = 'c:/datasets/0804-Parking_Dataset_coco/valid.json'
cfg.data.val.img_prefix = 'c:/datasets/0804-Parking_Dataset_coco/images/'

# Set parameters for Test dataset
cfg.data.test.type = 'ParkingDataset'
cfg.data.test.ann_file = 'c:/datasets/0804-Parking_Dataset_coco/test.json'
cfg.data.test.img_prefix = 'c:/datasets/0804-Parking_Dataset_coco/images/'
#print(cfg)

# Set number of classes
cfg.model.roi_head.bbox_head.num_classes = 14

# Set anchors: Reduce size
cfg.model.rpn_head.anchor_generator.scales = [4]

# Set a path of pretrained model
cfg.load_from = './_0804_parking_dataset_dynamic_rcnn_r50_fpn_1x-62a3f276.pth'

# Set a path for root directory
cfg.work_dir = './work_dirs/0804_Parking_dataset/'



# Set configuration parameters
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 6
cfg.checkpoint_config.interval = 1

# Set epochs
cfg.seed = 85
cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
set_random_seed(0, deterministic=False)
#print(cfg.pretty_text)


# Set datasets
datasets = [build_dataset(cfg.data.train)]
#print(datasets[0])

datasets[0].__dict__.keys()

# Set models
model = build_detector(cfg.model,
                       train_cfg = cfg.get('train_cfg'),
                       test_cfg = cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
#print(model.CLASSES) # Result: ('세단(승용차)', 'SUV', '승합차', '버스', '학원차량(통학버스)', '트럭', '택시', '성인', '어린이', '오토바이', '전동킥보드', '자전거', '유모차', '쇼핑카트')
#exit()

# Run codes
if __name__ == '__main__':

    # Set a path for resume file
    #cfg.resume_from = './work_dirs/0807-Website-screenshot-dataset/latest.pth'

    train_detector(model, datasets, cfg, distributed=False, validate=True)



