# Import modules
import cv2
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# Define a function to process data by batch when using DataLoader
def collate_fn(batch):
    # Get lists of values from batch data
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    # print(images)
    # exit()

    # Combine all images into a single tensor by using 'torch.stack'
    images = torch.stack(images, 0)

    # Initialize a list of targets
    targets = []

    # Get bounding boxes
    for i in range(len(targets_boxes)):
        target = {
            'boxes': targets_boxes[i],
            'labels': targets_labels[i]
        }
        targets.append(target)

    return images, targets


# Define a main class
class CustomDataset(Dataset):
    # Initialize the class
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.path_image = sorted(glob.glob(os.path.join(root, '*.png')))
        #self.path_image = sorted(glob.glob(root + '/*.png'))
        #print(self.root)         # Result: './datasets/0725-Automobile Dataset/train'
        #print(self.path_image)  # Result: './datasets/0725-Automobile Dataset/train\\syn_06480.png'


        # Get labels
        if train:
            self.boxes = sorted(glob.glob(os.path.join(root, '*.txt')))
            #self.boxes = sorted(glob.glob(root + '/*.txt'))
        #print(self.boxes) # Result: ./datasets/0725-Automobile Dataset/train\\syn_06480.txt'

    # Define a function for parsing bounding boxes
    def parse_boxes(self, path_box):
        with open(path_box, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            #print(lines)

        # Initialize lists
        boxes = []
        labels = []

        # Get lines
        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            #print(values) # Result: [9.0, 1037.0, 209.0, 1312.0, 209.0, 1312.0, 448.0, 1037.0, 448.0]
                           #         [25.0, 804.0, 425.0, 1127.0, 425.0, 1127.0, 783.0, 804.0, 783.0]
                           #         [12.0, 330.0, 250.0, 583.0, 250.0, 583.0, 511.0, 330.0, 511.0]

            class_id = int(values[0])
            x_min = int(round(values[1]))
            y_min = int(round(values[2]))
            x_max = int(round(max(values[3], values[5], values[7])))
            y_max = int(round(max(values[4], values[6], values[8])))
            #print(x_min, y_min, x_max, y_max)  # Result: 1037 209 1312 448


            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)
            #print(boxes)  # Result: [[1037, 209, 1312, 448]]


        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int)



    def __getitem__(self, item):
        path_img = self.path_image[item]
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]
        #print(height, width)  # Result: 1040 1920


        if self.train:
            path_box = self.boxes[item]
            #print(path_box) # Result: ./datasets/0725-Automobile Dataset/train\syn_00000.txt


            boxes, labels = self.parse_boxes(path_box)
            #print(boxes, labels)  # Result: ['9.0 1037 209 1312 209 1312 448 1037 448\n', '25.0 804 425 1127 425 1127 783 804 783\n', '12.0 330 250 583 250 583 511 330 511\n']
            #exit()
            labels += 1  # Two stage: starting from '1' <-> One stage: starting from '0'

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes = transformed['image'], transformed['bboxes']
                labels = transformed['labels']

            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed['image']
            file_name = path_img.split('\\')[-1]
            #print(file_name)
            #exit()
            return file_name, img, width, height

    def __len__(self):
        return len(self.path_image)


# Run codes
if __name__ == '__main__':
    train_dataset = CustomDataset('./datasets/0725-Automobile Dataset/train', train=True, transforms=None)

    for i in train_dataset:
        print(i)