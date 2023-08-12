# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from ref_0810_mnist_model import Autoencoder

# Set hyperparameters
#batch_size = 248
#batch_size = 326
batch_size = 2048
lr = 0.005
num_epochs = 100

# Set augmentations
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# Set dataset
train_dataset = torchvision.datasets.MNIST(root = 'c:/datasets/0811-MNIST Dataset',
                                           train = True,
                                           transform = transforms,
                                           download = True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


# Load a model
autoencoder = Autoencoder().to('cuda')
criterion = nn.MSELoss().to('cuda')
optimizer = optim.AdamW(autoencoder.parameters(), lr = lr, weight_decay=1e-4)

# Train datasets
for epoch in range(num_epochs):
    # Set a point of start time
    start_time = time.time()

    # Load data
    for data in train_loader:
        # Get image data
        img, _ = data
        img = img.to('cuda')
        img = img.view(img.size(0), -1).to('cuda')  # 2D to 1D

        # Set optimizer and functions
        optimizer.zero_grad()
        outputs = autoencoder(img)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()

    # Set a point of end time
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'EPOCH [{epoch + 1:03d}/{num_epochs}], Loss: {loss.item():.4f}, Duration: {epoch_time:.2f} secs')

# Save as a file
torch.save(autoencoder.state_dict(), './outcomes/0811-mnist-autoencoder_adamw_lr0.005.pt')



