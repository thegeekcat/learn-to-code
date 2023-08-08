# Import modules
from mmcv import Config
from mmdet.apis import set_random_seed

# Set a path of config file
config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)
#print(cfg.pretty_text)


# Set optimizer
cfg.optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.0001)
#cfg.optimizer.lr = 0.01

# Set dataset
cfg.dataset_type = 'FlowChartDataset'
cfg.dataset_root = 'c:/datasets/0808-Flow chart detection_coco'

# Set Paths
cfg.data.train.type = 'FlowChartDataset'
cfg.data.train.ann_file = 'c:/datasets/0808-Flow chart detection_coco/train/_annotations.coco.json'
cfg.data.train.img_prefix = 'c:/datasets/0808-Flow chart detection_coco/train'

cfg.data.val.type = 'FlowChartDataset'
cfg.data.val.ann_file = 'c:/datasets/0808-Flow chart detection_coco/valid/_annotations.coco.json'
cfg.data.val.img_prefix = 'c:/datasets/0808-Flow chart detection_coco/valid'

cfg.data.test.type = 'FlowChartDataset'
cfg.data.test.ann_file = 'c:/datasets/0808-Flow chart detection_coco/test/_annotations.coco.json'
cfg.data.test.img_prefix = 'c:/datasets/0808-Flow chart detection_coco/test'
# print(cfg.pretty_text)
# exit()


# Set number of classes
cfg.model.roi_head.bbox_head.num_classes = 20
#"categories":[{"id":0,"name":"Flow-chart","supercategory":"none"},{"id":1,"name":"action","supercategory":"Flow-chart"},{"id":2,"name":"activity","supercategory":"Flow-chart"},{"id":3,"name":"commeent","supercategory":"Flow-chart"},{"id":4,"name":"control_flow","supercategory":"Flow-chart"},{"id":5,"name":"control_flowcontrol_flow","supercategory":"Flow-chart"},{"id":6,"name":"decision_node","supercategory":"Flow-chart"},{"id":7,"name":"exit_node","supercategory":"Flow-chart"},{"id":8,"name":"final_flow_node","supercategory":"Flow-chart"},{"id":9,"name":"final_node","supercategory":"Flow-chart"},{"id":10,"name":"fork","supercategory":"Flow-chart"},{"id":11,"name":"merge","supercategory":"Flow-chart"},{"id":12,"name":"merge_noode","supercategory":"Flow-chart"},{"id":13,"name":"null","supercategory":"Flow-chart"},{"id":14,"name":"object","supercategory":"Flow-chart"},{"id":15,"name":"object_flow","supercategory":"Flow-chart"},{"id":16,"name":"signal_recept","supercategory":"Flow-chart"},{"id":17,"name":"signal_send","supercategory":"Flow-chart"},{"id":18,"name":"start_node","supercategory":"Flow-chart"},{"id":19,"name":"text","supercategory":"Flow-chart"}]

# Set anchors: Reduce size
cfg.model.rpn_head.anchor_generator.scales = [4]

# Set a path of pretrained model
cfg.load_from = './_0808_flowchart_dataset_faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
#cfg.resume_from = './work_dirs/0808_Flowchart_dataset/latest.pth'


# Set a working directory
cfg.work_dir = 'work_dirs/0808_Flowchart_dataset/r50_1x_AdamW_lr-0.005_lrconfig-linear/'

# Set a learning schedule
#cfg.lr_config.warmup = None
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[6, 10],
    gamma=0.5)

# Set parameters
cfg.log_config.interval = 10  # Print out log in every 10 batch

# Set Metrix boxes from CocoDataset
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 1  # Evaluate every 5 epochs

# Save model
cfg.checkpoint_config.interval = 3  # Save log files every 3 epochs


# Set parameters
cfg.runner.max_epochs = 100
cfg.seed = 85
cfg.data.samples_per_gpu = 10
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
set_random_seed(0, deterministic=False)


#print(cfg.pretty_text)
