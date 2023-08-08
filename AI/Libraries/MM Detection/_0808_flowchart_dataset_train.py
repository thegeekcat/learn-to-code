# Import modules
from mmdet.datasets import build_dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from _0808_flowchart_dataset_config import cfg

@DATASETS.register_module(force=True)

# Define a class
class FlowChartDataset(CocoDataset):

    # Set labels
    CLASSES = ('Flow-chart', 'action', 'activity', 'commeent', 'control_flow',
               'control_flowcontrol_flow', 'decision_node', 'exit_node', 'final_flow_node', 'final_node',
               'fork', 'merge', 'merge_noode', 'null', 'object',
               'object_flow', 'signal_recept', 'signal_send', 'start_node', 'text'
    )


# Run codes
if __name__ == '__main__':
    datasets = [build_dataset(cfg.data.train)]
    #print(datasets)
    # Result:
    # FlowChartDataset Train dataset with number of images 1503, and instance counts:
    # +------------------------------+-------+--------------------+-------+------------------+-------+---------------------+-------+------------------+-------+
    # | category                     | count | category           | count | category         | count | category            | count | category         | count |
    # +------------------------------+-------+--------------------+-------+------------------+-------+---------------------+-------+------------------+-------+
    # | 0 [Flow-chart]               | 0     | 1 [action]         | 3591  | 2 [activity]     | 5106  | 3 [commeent]        | 330   | 4 [control_flow] | 16968 |
    # | 5 [control_flowcontrol_flow] | 36    | 6 [decision_node]  | 1794  | 7 [exit_node]    | 30    | 8 [final_flow_node] | 78    | 9 [final_node]   | 1422  |
    # | 10 [fork]                    | 630   | 11 [merge]         | 624   | 12 [merge_noode] | 486   | 13 [null]           | 6     | 14 [object]      | 201   |
    # | 15 [object_flow]             | 72    | 16 [signal_recept] | 108   | 17 [signal_send] | 75    | 18 [start_node]     | 1428  | 19 [text]        | 1854  |
    # +------------------------------+-------+--------------------+-------+------------------+-------+---------------------+-------+------------------+-------+]

    model = build_detector(cfg.model,
                           train_cfg = cfg.get('train_cfg'),
                           test_cfg = cfg.get('test_cfg'))
    #print(datasets[0].__dict__.keys()) # Result: dict_keys(['ann_file', 'data_root', 'img_prefix', 'seg_prefix', 'seg_suffix', 'proposal_file', 'test_mode', 'filter_empty_gt', 'file_client', 'CLASSES', 'coco', 'cat_ids', 'cat2label', 'img_ids', 'data_infos', 'proposals', 'flag', 'pipeline'])
    #print(datasets[0].data_infos) # Result: [{'id': 0, 'license': 1, 'file_name': '220_png_jpg.rf.439b649b1ef7a97f9c6f33ffde7940fd.jpg', 'height': 640, 'width': 640, 'date_captured': '2023-06-13T05:48:37+00:00', 'filename': '220_png_jpg.rf.439b649b1ef7a97f9c6f33ffde7940fd.jpg'},

    model.CLASSES = datasets[0].CLASSES
    #print(model.CLASSES)  # Result: ('Flow-chart', 'action', 'activity', 'commeent', 'control_flow', 'control_flowcontrol_flow', 'decision_node', 'exit_node', 'final_flow_node', 'final_node', 'fork', 'merge', 'merge_noode', 'null', 'object', 'object_flow', 'signal_recept', 'signal_send', 'start_node', 'text')


    # Run codes
    train_detector(model, datasets, cfg, distributed=False, validate=True)

