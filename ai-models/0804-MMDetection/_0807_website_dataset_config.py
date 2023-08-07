# Import modules
import mmcv
from mmcv import Config
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

# Set a path of config file
config_file = './configs/detr/detr_r50_8x2_150e_coco.py'
cfg = Config.fromfile(config_file)

# Set Learning Rate
cfg.optimizer.lr = 0.001

# Set a dataset
cfg.dataset_type = 'WebsiteScreenshotDataset'
cfg.dataset_root = 'c:/datasets/0807-Website Screenshots Dataset_coco'


# Set Train, Validation, and Test
cfg.data.train.type = 'WebsiteScreenshotDataset'
cfg.data.train.ann_file = 'c:/datasets/0807-Website Screenshots Dataset_coco/train/_annotations.coco.json'
cfg.data.train.img_prefix = 'c:/datasets/0807-Website Screenshots Dataset_coco/train/'

cfg.data.val.type = 'WebsiteScreenshotDataset'
cfg.data.val.ann_file = 'c:/datasets/0807-Website Screenshots Dataset_coco/valid/_annotations.coco.json'
cfg.data.val.img_prefix = 'c:/datasets/0807-Website Screenshots Dataset_coco/valid/'

cfg.data.test.type = 'WebsiteScreenshotDataset'
cfg.data.test.ann_file = 'c:/datasets/0807-Website Screenshots Dataset_coco/test/_annotations.coco.json'
cfg.data.test.img_prefix = 'c:/datasets/0807-Website Screenshots Dataset_coco/test/'

# Set number of classes
cfg.model.bbox_head.num_classes = 9


# Set a path of a pretrained model
cfg.load_from = './_0807_website_dataset_detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
#print(cfg.pretty_text)

# Set a working directory
cfg.work_dir = './work_dirs/0807-Website-screenshot-dataset/'

# Set parameters
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Set Metrix boxes from Coco Dataset
#  bbox -> mAP IOU Threshold: 0.0 - 0.95
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 5

# Save model
cfg.checkpoint_config.interval = 3

# Set parameters
cfg.runner.max_epochs = 100
cfg.seed = 85
cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

