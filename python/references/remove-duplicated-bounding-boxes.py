
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
