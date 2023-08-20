
# Import modules
import random
import numpy as np
import os
import torch
import torchvision
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
from ref_0725_automobile_customdataset import CustomDataset, collate_fn
from ref_0725_automobile_config import config

import datetime
import warnings  # To ingore warning messages
warnings.filterwarnings(action='ignore')

# Define a main function
def main():

    # Check device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #print('Device: ', device)

    # Fix Random-seed
    def fixed_seed(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # Set fixed seed
        fixed_seed(config['SEED'])
        # seed = fixed_seed(config['SEED'])
        # print(seed)
        # exit()



    # Define a function for augmentations for Train dataset
    def get_train_transforms():
        return A.Compose([
            A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
            ToTensorV2()
        ], bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels']))


    # Define a function for augmentations for Test dataset
    def get_test_transforms():
        return A.Compose([
            A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
            ToTensorV2()
        ])   # Reason there's no 'bbox_params' for test dataset
             #  : the Test dataset doesn't have text files for labels



    # Set Datasets
    dataset_train = CustomDataset('./datasets/0725-Automobile Dataset/train', train=True, transforms=get_train_transforms())
    dataset_test = CustomDataset('./datasets/0725-Automobile Dataset/test', train=False, transforms=get_test_transforms())
    #print(dataset_test)

    # Test
    # for i in dataset_test:
    #     print(i)

    # Set DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=config['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=config['BATCH_SIZE'], shuffle=False)
    #print(dataloader_test)


    # Define a function for models
    def build_model(num_classes=config['NUM_CLASS'] + 1):  # Reason of adding '1': Two stage model starts from '0' in PyTorch
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        #print(model)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        return model


    # Load model on device
    model = build_model()
    model.to(device)


    # Define a function for Training a model
    def train(model, dataloader_train, optimizer, scheduler, device, resume_checkpoint=None):
        # Set parameters
        best_loss = 999999
        start_epoch = 1

        # Set Resume function
        if resume_checkpoint is not None:
            checkpoint = torch.load(resume_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resuming training from epoch {start_epoch}!')

        # Set epochs
        for epoch in (range(start_epoch, config['EPOCHS'] + 1)):

            # Set mode as train
            model.train()

            # Initialize parameters
            train_loss = []

            # Set batches
            num_batches = len(dataloader_train)

            #
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader_train,
                                                               total=num_batches,
                                                               desc=f'Epoch [{epoch}] Batches',
                                                               leave=True,
                                                               mininterval=0)):

                # Get images and labels
                images = [img.to(device) for img in images]
                targets = [{k:v.to(device) for k, v in t.items()} for t in targets]

                # Initialize optimizer
                optimizer.zero_grad()

                # Get losses
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backpropagation
                losses.backward()
                optimizer.step()

                train_loss.append(losses.item())

            # Calculate Train Loss
            tr_loss = np.mean(train_loss)
            tqdm.write(f'Epoch [{epoch} Train Loss: {tr_loss:.5f}')

            # Set Scheduler
            if scheduler is not None:
                scheduler.step()

            # Set the current time
            now = datetime.datetime.now()
            ## Need to change type to 'string'
            ##   : 'now.strftime('%Y_%m_%d')

            ## Reference: Make a folder by time
            # def make_dir_by_time(self) :
            #         now = datetime.now()
            #         now = str(now)
            #         now = now.split(".")[0]
            #         now = now.replace("-","").replace(" ","_").replace(":","")
            #         self.result_dir = os.path.join("./result2", now)
            #         os.makedirs(self.result_dir, exist_ok=True)

            # Get the best model
            if best_loss > tr_loss:
                best_loss = tr_loss
                best_model = model.state_dict()
                torch.save(best_model, f'./outcomes/0725-automobile-bestmodel_lr0.001_10epochs.pt')

            # Set checkpoint to save progress
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint, f'./outcomes/0725-automobile-checkpoint_lr0.001_10epochs.pt')

    # Set Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    ## Reference: How to get the name of optimizer?
    ##  1. 'type(optimizer).__name__'
    ##  2. Get inputs from users

    # Run codes
    train(model, dataloader_train, optimizer, scheduler, device, resume_checkpoint=None)





# Run codes
if __name__ == '__main__':
    main()

