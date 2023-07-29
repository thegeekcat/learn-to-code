# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from ref_0714_customdata_cottonweed import CustomDataset_CottonWeed

from tqdm import tqdm
import cv2

# Set Label Dictionary
label_dict = {0: 'Chickenpox', 1: 'Cowpox', 2: 'Healthy', 3: 'HFMD', 4: 'Measles', 5: 'Monkeypox'}


# Define Main class
def main():
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # Set Models
    model = mobilenet_v2(pretrained=False)
    # print(model)
    in_features_ = 1280
    model.classifier[1] = nn.Linear(in_features_, 6)
    # print(model)

    # Load Model
    model.load_state_dict(torch.load(f='./outcomes/0714-Skin Diseases_from_teacher.pt'))
    print(list(model.parameters()))

    # Set transforms
    transforms_validation = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    # Set Dataset Path
    dataset_validation = CustomDataset_CottonWeed('./outcomes/0714-SkinDiseases/Validation',
                                                  transforms=transforms_validation)
    dataloader_validation = DataLoader(dataset_validation, batch_size=1, shuffle=False)

    # Set Mode in Evaluation
    model.to(device)
    model.eval()

    # Initialize Accuracy Score
    accuracy_score = 0

    # Display the result
    with torch.no_grad():
        for data, target, path in tqdm(dataloader_validation):
            # Search dictionary
            target_ = target.item()

            # Get data and labels
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)
            # exit()

            # Set Prediction
            prediction = output.argmax(dim=1, keepdim=True)
            # print(prediction)  # Result: tensor([[0]], device='cuda:0')
            # exit()

            # Set Labels
            label_target = label_dict[target_]
            label_predicted = label_dict[prediction.item()]
            # print('Answer Labels: ', prediction.item(), label_target)   # Result: 0 Carpetweeds
            # print('Predicted Labels: ', prediction.item(), label_predicted)
            # exit()

            # Get Label Texts
            label_text_answer = f'Answer labels: {label_target}'
            label_text_predicted = f'Predicted labels: {label_predicted}'
            # print(path)  # Result: ('./outcomes/0714-Cotton Weed ID15/Validation\\Carpetweeds\\Carpetweeds_321.jpg',)
            # exit()

            # Visualization
            image = cv2.imread(path[0])
            image = cv2.resize(image, (500, 500))
            image = cv2.rectangle(image, (0, 0), (500, 100), (255, 255))
            image = cv2.putText(image, label_text_predicted, (0, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            image = cv2.putText(image, label_text_answer, (0, 75), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            # print(image)
            cv2.imshow('Test', image)
            if cv2.waitKey() == ord('q'):
                exit()

            # Get Accuracy Scores
            accuracy_score += prediction.eq(target.view_as(prediction)).sum().item()

        # Display Accuracy Scores
    print('Test Result of Accuracy Scores: {}/{} [{:.0f}]%\n'.format(accuracy_score, len(dataloader_validation.dataset),
                                                                     100 * accuracy_score / len(
                                                                         dataloader_validation.dataset)))


# Run the codes
if __name__ == '__main__':
    main()






