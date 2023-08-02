# Import modules
import numpy as np
import cv2
import matplotlibpyplot as plt
import matplotlib.patches as patches
import torchvision
import torch
from torchvision.models.detection.rpn import AnchorGenerator
from utils import collate_fn
from torch.utils.data import DataLoader
from ref_0801_glue-tube-customdataset import KeypointDataset

import warnings
warnings.filterwarnings(action='ignore')
# Define a function to remove duplicated boxes
def filter_duplicate_boxes(boxes, iou_treshold=0.5):  # 0.5 - 0.6 is a normal range of threshold for IoU
    # Initialize a list of boxes after removing duplications
    boxes_filtered = []

    for i, box1 in enumerate(boxes):
        is_duplicate = False
        for j, box2 in enuerate(boxes_filtered):
            print(f'IoU:  {calculate_iou(box1, box2)}')
            if calculate_iou(box1, box2) > iou_treshold:
                is_duplicate = True
                # A code is addable to remove duplicated bbox
                break
            if not is_duplicate:
                boxes_filtered.append(box1)
    return boxes_filtered


# Define a main function
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    keypoint_test_data_path = './val' # NEED TO FIX
    test_dataset = KeypointDataset(keypoint_test_data_path,
                                   transform = None,
                                   demo = False)
    test_loader = DataLoader(test_dataset,
                             batch_size = 1,
                             shuffle = False,
                             collate_fn = collate_fn)
    iterator = inter(test_loader)

    # Set model
    model = get_model(num_keypoints = 2)
    model.load_state_dict(torch.load(f='./0801-keypointsrcnn_weights_20.pth',  # NEED TO FIX
                                     map_location = device))  # 'map_location=device': Relocate model between 'cpu' and 'gpu'

    model.to(device)
    model.eval()

    # Handle duplicated areas
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            output = model(images)
            # print(output)
            # exit
        #
        image = (images[0].permute(1, 2, 0), detach().cpu().numpy()*255).astype(np.uint8)
        scores = output[0]['scores'].detach.copu().numpy()

        # Calculate indexes of the highest score
        high_scores_idx = np.where(scores>0.7)[0].tolist()

        # Calculate nms
        post_nms_idx = (torchvision.ops.nms(output[0]['boxes'][high_scores_idx],
                                           output[0]['scores'][high_scores_idx], 0.3)
                                        .cpu().numpy)
        #print('post_nms_idxs: ', post_nms_idx)
        post_nms_boxes = output[0]['boxes'][high_scores_idx][post_nms_idx].detach().cpu().numpy
        filtered_boxes = filter_duplicate_boxes(post_nms_boxes, iou_treshold=0.1)
        print(filtered_boxes)