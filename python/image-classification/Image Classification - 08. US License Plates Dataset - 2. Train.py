# Import module
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import albumentations as A
import argparse

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_b0
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from ref_0718_customdataset_uslicenseplates import MyUSLicensePlatesDataset



# Define a main
def main():
    pass

# Define a class
class US_LicensePlate_Classifier:
    # Initialize the class
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.losses_train = []
        self.losses_validation = []
        self.accuraccies_train = []
        self.accuraccies_validation = []

    # Define train
    def train(self, dataloader_train, dataloader_validation, epochs, optimizer, criterion, start_epoch=0):
        best_accuracy_validation = 0.0
        print('Training.....')

        # Fit models
        for epoch in range(start_epoch, epochs):
            # Initialize
            loss_train = 0.0
            loss_validation = 0.0
            accuracy_train = 0.0
            accuracy_validation = 0.0

            # Change mode in Train mode
            self.model.train()
            interator_dataloader_train = tqdm(dataloader_train, desc=(f'Epoch: {epoch+1}/{epochs}'), leave=True)

            for index, (data, target) in enumerate(interator_dataloader_train):
                # Get data, labels
                data, target = data.float().to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                # Calculate Train Loss
                loss_train += loss.item()

                # Get Prediction
                _, prediction = torch.max(outputs, 1)

                # Calculate accuracy of train
                accuracy_train += (prediction == target).sum().item()

                # Get Loss values
                interator_dataloader_train.set_postfix({'Loss': loss.item()})

            # Calculate Train Loss
            loss_train /= len(dataloader_train)
            accuracy_train = accuracy_train / len(dataloader_train.dataset)

            ###### Set Evaluation Part #####
            # Set a mode in evaluation
            self.model.eval()

            # Fit models for evaluation
            with torch.no_grad():
                for data, target in dataloader_validation:
                    # Get data and labels
                    data, target = data.float().to(self.device), target.to(self.device)
                    output = self.model(data)
                    prediction = output.argmax(dim=1, keepdim=True)
                    accuracy_validation += prediction.eq(target.view_as(prediction)).sum().item()
                    loss_validation += criterion(output, target).item()

            # Calculate Loss and Accuracy for Evaluation
            loss_validation /= len(dataloader_validation)
            accuracy_validation = accuracy_validation / len(dataloader_validation.dataset)


            ###### Get for all #####
            # Get lists for Loss and Accuracy
            self.losses_train.append(loss_train)
            self.accuraccies_train.append(accuracy_train)
            self.losses_validation.append(loss_train)
            self.accuraccies_validation.append(accuracy_validation)

            # Display results
            print(f'Epoch [{epoch + 1} / {epochs}]: ,'
                  f'Loss: Train [{loss_train:.4f}], Loss: Validation [{loss_validation:.4f}], '
                  f'Accuracy: Train [{accuracy_train:.4f}], Accuracy: Validation [{accuracy_validation:.4f}]')

            # Calculate Best Accuracy of Validation and save as 'pt' files
            if accuracy_validation > best_accuracy_validation:
                torch.save(self.model.state_dict(), './outcomes/0718-US_license_plates_efficientnet_b0_best.pt')

            # Save Model Status and Optimizer Status per each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses_train': self.losses_train,
                'accuraccies_train': self.accuraccies_train,
                'losses_validation': self.losses_validation,
                'accuraccies_validation': self.accuraccies_validation
            }, './outcomes/weight/0718-US_license_plates_efficientnet_b0_checkpoint.pt')

        # Save file as 'pt' type
        torch.save(self.model.state_dict(), './outcomes/0718-US_license_plates_efficientnet_b0_last.pt')

        # Initalize csv and plots
        self.save_result_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    # Define Save Result to CSV files
    def save_result_to_csv(self):
        df = pd.DataFrame({
            'Loss: Train': self.losses_train,
            'Accuracy: Train': self.accuraccies_train,
            'Loss: Validation': self.losses_validation,
            'Accuracy: Validation': self.accuraccies_validation
        })
        df.to_csv('./outcomes/result_0718-US_license_plates_efficientnet_b0.csv', index=False)

    # Define Display for Loss
    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses_train, label='Train Loss')
        plt.plot(self.losses_validation, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./outcomes/visualization_0718-US_license_plates_efficientnet_b0_loss.jpg')

    # Define display for Accuracy
    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.accuraccies_train, label='Train Accuracy')
        plt.plot(self.accuraccies_validation, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./outcomes/visualization_0718-US_license_plates_efficientnet_b0_accuracy.jpg')


    # Run models
    def run(self, args):
        self.model = efficientnet_b0(pretrained=True)
        #print(self.model)  # Result:   (classifier): Sequential(
                            #                                    (0): Dropout(p=0.2, inplace=True)
                            #                                    (1): Linear(in_features=1280, out_features=1000, bias=True))
        #exit()

        # Modify models
        self.model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[1] = nn.Linear(1280, out_features=50)
        #print(self.model)
        self.model.to(self.device)

        # Set augmentations
        transforms_train = A.Compose([
            A.Resize(width=224, height=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomShadow(),
            A.RandomRain(),
            A.RandomFog(),
            A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.3),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomBrightness(),
            A.RandomRotate90(),
            A.RandomGamma(),
            ToTensorV2()
        ])
        transforms_validation = A.Compose([
            A.Resize(width=224, height=224),
            ToTensorV2()
        ])

        # Set Dataset
        dataset_train = MyUSLicensePlatesDataset(args.dir_train, transform=transforms_train)
        dataset_validation = MyUSLicensePlatesDataset(args.dir_validation, transform=transforms_validation)

        # Set DataLoader
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_validation = DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=True)

        # Set Loss and Optimizer
        epochs = args.epochs
        criterion = CrossEntropyLoss().to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        start_epoch = 0

        # Set Resume function
        if args.resume_training:
            checkpoint = torch.load(args.path_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.losses_train = checkpoint['losses_train']
            self.accuraccies_train = checkpoint['accuraccies_train']
            self.losses_validation = checkpoint['losses_validation']
            self.accuraccies_validation = checkpoint['accuraccies_validation']

        self.train(dataloader_train, dataloader_validation, epochs, optimizer, criterion, start_epoch=start_epoch)



# Run codes
if __name__ == '__main__':
    # Set Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_train', type=str, default='./data/0718-US License Plates Dataset/Train',
                        help='A directory path to the training dataset')
    parser.add_argument('--dir_validation', type=str, default='./data/0718-US License Plates Dataset/Validation',
                        help='A directory path to the validation dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=124,
                        help='A batch size for training and validation')
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help='weight decay for optimizer')
    parser.add_argument("--resume_training", action='store_true',
                        help='resume training from the last checkpoint')
    parser.add_argument("--path_checkpoint", type=str,
                        default="./outcomes/weight/0718-US_license_plates_efficientnet_b0_checkpoint.pt",
                        help="path to the checkpoint file")
    parser.add_argument("--path_folder_checkpoint", type=str,
                        default="./outcomes/weight/")

    args = parser.parse_args()

    # Set a folder path for weight
    path_folder_weight = args.path_folder_checkpoint
    os.makedirs(path_folder_weight, exist_ok=True)

    classifier = US_LicensePlate_Classifier()
    classifier.run(args)


    # Resume:
    # $ python ./Image Classification - 08. US License Plates Dataset - 2. Train.py --resume_training
