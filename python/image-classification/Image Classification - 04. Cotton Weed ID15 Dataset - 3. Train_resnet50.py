# Import modules
import csv

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.resnet import resnet50
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from lion_pytorch import Lion

from ref_0714_customdata_cottonweed import CustomDataset_CottonWeed


# Define Train Model
def train(model, dataloader_train, dataloader_validation, epochs, optimizer, criterion, device):
    # Initialize parameters
    best_val_acc = 0.0
    losses_train = []
    losses_validation = []
    accuracies_train = []
    accuracies_validation = []

    # Fit model
    print('\n\n\nTraining the model......')
    for epoch in range(epochs):

        # Initailize parameters
        loss_train = 0.0
        loss_validation = 0.0
        accuracy_train = 0.0
        accuracy_validation = 0.0


        # Set in Train mode
        model.train()


        # Set
        loader_iterater_train = tqdm(dataloader_train,
                                     desc=(f'Epoch: {epoch+1}/{epochs}'),
                                     leave=True)   # 'leave=True': Keep displaying progress bar after learning is done

        # Fit model
        for i, (data, target) in enumerate(loader_iterater_train):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            # Set loss
            loss = criterion(outputs, target)
            loss.backward()

            # Do backpropagation
            optimizer.step()

            # Calculate Train Loss
            loss_train += loss.item()

            # Calculate Accuracy
            _, prediction = torch.max(outputs, 1)
            accuracy_train += (prediction == target).sum().item()

            # Display the result
            loader_iterater_train.set_postfix({'Loss: ': loss.item()})

        # Calculate Loss and Accuracy for Train
        loss_train /= len(dataloader_train)
        accuracy_train = accuracy_train / len(dataloader_train.dataset)


        # Set Evaluation
        model.eval()
        with torch.no_grad():
            for data, target in dataloader_validation:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)

                # Calculate accuracy and loss
                accuracy_validation += prediction.eq(target.view_as(prediction)).sum().item()
                loss_validation += criterion(output, target).item()


        # Calculate loss and accuracy for Validation set
        loss_validation /= len(dataloader_validation)
        accuracy_validation = accuracy_validation / len(dataloader_validation.dataset)


        # Get results
        losses_train.append(loss_train)
        accuracies_train.append(accuracy_train)
        losses_validation.append(loss_validation)
        accuracies_validation.append(accuracy_validation)


        # Save models as a 'csv' file
        data_csv = []
        for epoch, loss_train, loss_validation, accuracy_train, accuracy_validation in zip (epochs, losses_train, losses_validation, accuracies_train, accuracies_validation):
            data_csv.append({'Epoch': epoch, 'Loss: Train': loss_train, 'Loss: Validation': loss_validation, 'Accuracy: Train': accuracy_train, 'Accuracy: Validation': accuracy_validation})
        fieldnames = ['Epoch', 'Loss: Train', 'Loss: Validation', 'Accuracy: Train', 'Accuracy: Validation']
        filename = 'result__0714__data-cottonweed__model-mobilenet_v2__loss-CrossEntropy__optim-AdamW.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerheader()
            writer.writerows(data_csv)


        # Save models  as a 'pt' file
        if accuracy_validation > best_val_acc:
            torch.save(model.state_dict(), './outcomes/0714-Cotton Weed ID15_resnet50.pt')
            best_val_acc = accuracy_validation
        print(f'Epoch [{epoch+1} / {epochs}]: ,' 
              f'Loss: Train [{loss_train:.4f}], Loss: Validation [{loss_validation:.4f}], '
              f'Accuracy: Train [{accuracy_train:.4f}], Accuracy: Validation [{accuracy_validation:.4f}]')

    # Save the last model as a 'pt' file
    torch.save(model.state_dict(), './outcomes/0714-Cotton Weed ID15_resnet50_best.pt')
    return model, losses_train, losses_validation, accuracies_train, accuracies_validation



# Define a  main class
def main():
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Device: ', device)

    # Set model
    model = resnet50(pretrained=True)
    #print('Model: ', model)
    in_features_ = 2048
    model.fc = nn.Linear(in_features_, 15)
    #print('Modified Model: ', model)
    #model.load_state_dict(torch.load(f='./outcomes/')

    # Set device
    model.to(device)

    # Set Augmentations
    transforms_train = transforms.Compose([
        transforms.CenterCrop((244, 244)),
        transforms.Resize((244, 244)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transforms_validation = transforms.Compose([
        transforms.CenterCrop((244, 244)),
        transforms.Resize((244, 244)),
        transforms.ToTensor()
    ])

    # Set Dataset
    dataset_train = CustomDataset_CottonWeed('./outcomes/0714-Cotton Weed ID15/Train', transforms=transforms_train)
    dataset_validation = CustomDataset_CottonWeed('./outcomes/0714-Cotton Weed ID15/Validation', transforms=transforms_validation)
    #for i in dataset_train:
    #    print(i)


    # Set DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=126, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=126, shuffle=False, num_workers=4, pin_memory=True)
    #print('dataloader_validation: ', dataloader_validation)

    # Set Loss Function and Optimizer
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

    train(model, dataloader_train, dataloader_validation, epochs, optimizer, criterion, device)


# Run codes
if __name__ == '__main__':
    main()