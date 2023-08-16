# Import modules
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import albumentations as A
import cv2

from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from ref_0717_customdataset_food import MyFoodDataset


# Define Train
def train(model, dataloader_train, dataloader_validation, epochs, optimizer, criterion, device):
    # Initialize parameters
    best_accuracy_validation = 0.0
    losses_train = []
    losses_validation = []
    accuracies_train = []
    accuracies_validation = []
    print('Training.....')

    # Fit models
    for epoch in range(epochs):
        # Set initial values
        loss_train = 0.0
        loss_validation = 0.0
        accuracy_train = 0.0
        accuracy_validation = 0.0

        # Set a training mode
        model.train()

        # Fit models for Train
        iteration_train = tqdm(dataloader_train, desc=(f'Epoch: {epoch+1}/{epochs}'), leave=False)
        for i, (data, target) in enumerate(iteration_train):
            # Set devices
            data, target = data.float().to(device), target.to(device)

            # Set models
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # Calculate Train Loss
            loss_train += loss.item()

            # Calculate Train Accuracy
            _, prediction = torch.max(outputs, 1)
            accuracy_train += (prediction == target).sum().item()

            # Set
            iteration_train.set_postfix({'Loss': loss.item()})

        # Calculate Loss and Accuracy for Train
        loss_train /= len(dataloader_train)
        accuracy_train = accuracy_train / len(dataloader_train.dataset)


    # Fit models for Evaluation
    model.eval()
    with torch.no_grad():
        for data, target in dataloader_validation:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Calculate Validation Accuracy
            prediction = torch.max(output, 1)
            accuracy_validation += (prediction == target).sum().item()

    # Calculate Loss and Accuracy for Validation
    loss_validation /= len(dataloader_validation)
    accuracy_validation = accuracy_validation / len(dataloader_validation.dataset)

    # Get lists of Loss and Accuracy
    losses_train.append(loss_train)
    accuracies_train.append(accuracy_train)
    losses_validation.append(loss_validation)
    accuracies_validation.append(accuracy_validation)

    # Save models as a 'pt' file
    if accuracy_validation > best_accuracy_validation:
        torch.save(model.state_dict(), './outcomes/0717-food_mobilenet_v2_best.pt')

    # Display results
    print(f'Epoch [{epoch+1} / {epochs}]: ,' 
            f'Loss: Train [{loss_train:.4f}], Loss: Validation [{loss_validation:.4f}], '
            f'Accuracy: Train [{accuracy_train:.4f}], Accuracy: Validation [{accuracy_validation:.4f}]')

    # Save the last model as a 'pt' file
    torch.save(model.state_dict(), './outcomes/0717-food_mobilenet_v2_final.pt')

    # Save results of Train and Validation as a 'csv' file
    df = pd.DataFrame({
        'Train_Loss: ': losses_train,
        'Train_Accuracy': accuracies_train,
        'Validation_Loss': losses_validation,
        'Validation_Accuracy': accuracies_validation
    })
    df.to_csv('./outcomes/0717-food_mobilenet_v2', index=False)

    # Visualize and Save results of Loss and Accuracy of Train and Validation
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_validation, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outcomes/0717-food_mobilenet_v2_plot_loss.png')
    plt.figure()
    plt.plot(accuracies_train, label='Train Accuracy')
    plt.plot(accuracies_validation, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./outcomes/0717-food_mobilenet_v2_plot_accuracy.png')




# Define a main class
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Device: ', device)

    # Set model
    model = mobilenet_v2(pretrained=True)
    #print(model)
    model.classifier[1] = nn.Linear(1280, 20)
    model.to(device)

    # Augmentation with albumentations
    transforms_train = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.05,
                           rotate_limit=15,
                           p=0.6),  # 'p=0.6': 60%
        A.RandomShadow(),
        A.RGBShift(r_shift_limit=15,
                   g_shift_limit=15,
                   b_shift_limit=15,
                   p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])
    transforms_validation = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])


    # Set Datasets
    dataset_train = MyFoodDataset('./data/0717-Food Dataset/Train', transform=transforms_train)
    dataset_validation = MyFoodDataset('./data/0717-Food Dataset/Validation', transform=transforms_validation)

    # Set DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=124, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=124, shuffle=False)


    # Set Loss Function
    epoch = 20
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    train(model, dataloader_train, dataloader_validation, epoch, optimizer, criterion, device)


# Run codes
if __name__ == '__main__':
    main()