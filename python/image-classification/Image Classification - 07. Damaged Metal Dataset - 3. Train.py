# Import modules
import torch
import torch.nn as nn
import torchvision
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_b0
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from ref_0717_customdataset_damagedmetal import MyDataset


# Define a Train function
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
        iteration_train = tqdm(dataloader_train, desc=(f'Epoch: {epoch+1}/{epochs}'), leave=True)
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
        torch.save(model.state_dict(), './outcomes/0717-Damaged Metal Dataset_efficientnet_b0_best.pt')

    # Display results
    print(f'Epoch [{epoch+1} / {epochs}]: ,' 
            f'Loss: Train [{loss_train:.4f}], Loss: Validation [{loss_validation:.4f}], '
            f'Accuracy: Train [{accuracy_train:.4f}], Accuracy: Validation [{accuracy_validation:.4f}]')

    # Save the last model as a 'pt' file
    torch.save(model.state_dict(), './outcomes/0717-Damaged Metal Dataset_efficientnet_b0_final.pt')

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
    plt.savefig('./outcomes/0717-Damaged Metal Dataset_efficientnet_b0_plot_loss.png')
    plt.figure()
    plt.plot(accuracies_train, label='Train Accuracy')
    plt.plot(accuracies_validation, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./outcomes/0717-Damaged Metal Dataset_efficientnet_b0_plot_accuracy.png')


# Define a main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Device: ', device)

    # Set model
    model = efficientnet_b0(pretrained=True)
    #print(model) # Result: (classifier): Sequential(
                 #                                  (0): Dropout(p=0.2, inplace=True)
                 #                                  (1): Linear(in_features=1280, out_features=1000, bias=True))
    model.classifier = nn.Linear(1280, 10)
    model.to(device)
    #print(model)  # Result: (classifier): Linear(in_features=1280, out_features=10, bias=True)


    # Set transforms
    transforms_train = A.Compose([
        A.Resize(width=255, height=255),
        A.RandomShadow(),
        A.RandomBrightnessContrast(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        ToTensorV2()
    ])
    transforms_validation = A.Compose([
        A.Resize(width=255, height=255),
        ToTensorV2()
    ])

    # Set Datasets
    dataset_train = MyDataset('./outcomes/0717-Damaged Metal Dataset_cropped/Train', transform=transforms_train)
    dataset_validation = MyDataset('./outcomes/0717-Damaged Metal Dataset_cropped/Validation', transform=transforms_validation)

    # Set DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=124, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=124, shuffle=True)


    # Set Loss Function and Optimization
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)  # if use 'SGD', lr=0.1

    train(model, dataloader_train, dataloader_validation, epochs, optimizer, criterion, device)



# Run codes
if __name__ == '__main__':
    main()