# Import modules
import torch
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.runner import get_dist_info, init_dist
from mmdet.utils import  collect_env, get_root_logger, setup_multi_processes

from _0807_website_dataset_config import cfg

@DATASETS.register_module(force=True)

# Define a class
class WebsiteScreenshotDataset(CocoDataset):
    # Set labels
    CLASSES = ('elements','button', 'field', 'heading', 'iframe',
               'image', 'label', 'link', 'text')

# Run codes
if __name__ == '__main__':
    datasets = [build_dataset(cfg.data.train)]
    #print(datasets[0])

    model = build_detector(cfg.model,
                           train_cfg = cfg.get('train_cfg'),
                           test_cfg = cfg.get('test_cfg'))
    #print(datasets[0].__dict__.keys()) # Result: dict_keys(['ann_file', 'data_root', 'img_prefix', 'seg_prefix', 'seg_suffix', 'proposal_file', 'test_mode', 'filter_empty_gt', 'file_client', 'CLASSES', 'coco', 'cat_ids', 'cat2label', 'img_ids', 'data_infos', 'proposals', 'flag', 'pipeline'])
    #print(datasets[0].data_infos) # [{'id': 0, 'license': 1, 'file_name': 'forum_xda-developers_com_png.rf.00d6ce099ac81ed5846f08dad1e1e073.jpg', 'height': 768, 'width': 1024, 'date_captured': '2021-06-16T10:38:20+00:00', 'filename': 'forum_xda-developers_com_png.rf.00d6ce099ac81ed5846f08dad1e1e073.jpg'},

    model.CLASSES = datasets[0].CLASSES
    #print(model.CLASSES) # ('elements', 'button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text')

    # Set a path for resume file
    cfg.resume_from = 'work_dirs/0807-Website-screenshot-dataset/latest.pth'
    #cfg.resume_from = 'work_dirs/0807-Website-screenshot-dataset/epoch_12.pth'

    train_detector(model, datasets, cfg, distributed=False, validate=True)


