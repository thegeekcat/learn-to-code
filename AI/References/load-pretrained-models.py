# Import modules
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.models import resnet18
from ref_0713_customdata import CustomDataset_ex02


# Define main class
def main():
    # Set device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model
    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # Load Pre-trained model
    model.load_state_dict(torch.load(f='./0714-art_paintings.pt'))
    print(list(model.parameters()))

    pass

# Run
if __name__ == '__main__':
    main()