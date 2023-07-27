# Import modules
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.models import resnet18
from ref_0713_customdata import CustomDataset_ex02

from tqdm import tqdm
import cv2


# Define main class
def main():
    # Set device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Device: ', device)

    # Set model

    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # Load Pre-trained model
    model.load_state_dict(torch.load(f='./0714-art_paintings.pt'))
    #print(list(model.parameters()))

    # Set transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Get Test Datasets
    test_dataset = CustomDataset_ex02('./data/0713-splited-data_paintings/train', val_transforms)
    #print(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #print(test_loader)

    # Load model on device
    model.to(device)


    # Set model in evaluation mode
    model.eval()

    correct = 0

    # Set labels
    label_dict = {0: "Abstract",
                  1: "Cubist",
                  2: "Expressionist",
                  3: "Impressionist",
                  4: "Landscape",
                  5: "Pop Art",
                  6: "Portrait",
                  7: "Realist",
                  8: "Still Life",
                  9: "Surrealist"}

    # Get data and labels
    with torch.no_grad():
        for data, target, path in tqdm(test_loader):  # 'tqdm': Add progress bar
            # Get labels for answers
            target_ = target.item()

            # Load on 'device'
            data, target = data.to(device), target.to(device)

            # Set output
            output = model(data)
            print('Predicted Similarity Scores: ', output)
            #exit()
            ## Result: 'tensor([[ 9.7603, -5.3640, -1.2866, -3.0412,  1.5927, -4.9969, -1.1412, -4.4847, -5.2799, -4.6056]], device='cuda:0')'
            ##       -> Nine values are predicted similarity scores per each label


            # Get Prediction
            pred = output.argmax(dim=1, keepdim=True)


            # Display paths of images
            #print('Predction: ', pred.item(), path)   # Check

            ##### Visualization #####
            #print(type(path[0]))
            img = cv2.imread(path[0])
            img = cv2.resize(img, (500,500))

            # Set labels for predictions
            pred_label = label_dict[pred.item()]
            #print(type(pred.item()))
            pred_text = f'Prediction: {pred_label}'
            print(pred_text)

            # Get labels for answers
            target_label = label_dict[target_]
            #print(target_label)
            #exit()
            target_text = f'Correct Answer: {target_label}'
            print(target_text)


            # Display images
            img = cv2.rectangle(img, (0, 0), (500, 100), (255, 255, 255), -1)
            img = cv2.putText(img,             # Image file
                              pred_text,       # Text
                              (80,30),         # Text Location
                              cv2.FONT_ITALIC, # Font
                              1,               # Font Scale
                              (0, 0, 0),       # Color
                              1)               # Thickness
            img = cv2.putText(img, target_text, (80,75), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            cv2.imshow('Test', img)
            if cv2.waitKey() == ord('q'):
                exit()

            ###### Get correct score #####
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test Set: Accuracy {}/{} [{:.0f}%]\n'.format(correct, len(test_loader.dataset), 100*correct/len(test_loader.dataset)))

    pass

# Run
if __name__ == '__main__':
    main()