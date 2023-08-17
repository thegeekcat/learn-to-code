# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from ref_0811_mnist_model import DenoisingAutoEncoder


# Set parameters
batch_size = 246
lr = 0.001
num_epochs = 50
latent_dim = 20

# Set augmentations
transforms = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# Set dataset
train_dataset = torchvision.datasets.MNIST(root = 'c:/datasets/0811-MNIST Dataset',
                                           train = True,
                                           transform = transforms,
                                           download = False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

# Define a function to add noise
def add_noise(image, noise_factor=0.5):
    noisy_images = image + noise_factor * torch.rand_like(image)
    noisy_images = torch.clamp(noisy_images, -1, 1)

    return noisy_images

# Load a model
model = DenoisingAutoEncoder().to('cuda')  # Reason of using '.to('cuda')': To increase speed as all tensors and models are running on the same device

# Set Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Load models
for epoch in range(num_epochs):
    # Set a point of start time
    start_time = time.time()

    for i, (image, _) in enumerate(train_loader):
        noisy_images = add_noise(image, noise_factor=0.3)
        image = image.to('cuda')
        noisy_images = noisy_images.to('cuda')

        optimizer.zero_grad()
        outputs = model(noisy_images)

        loss = criterion(outputs.view(-1, 784),
                         image.view(-1, 784))
        loss.backward()
        optimizer.step()

    # Set a point of end time
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'Epoch: [{epoch + 1:02d} / {num_epochs}], Loss: {loss.item():.4f}, Duration: {epoch_time:.2f} secs')


# Save models as a file
torch.save(model.state_dict(), './outcomes/0811-mnist-denoising-autoencoder_lr0.001.pt')




