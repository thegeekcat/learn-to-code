# Import modules
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ref_0814_mnist_vae_model import VAE

# Set parameters
batch_size = 246
learning_rate = 0.005
latent_dim = 20
num_epochs = 150


# Set augmentations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, ), (1,))
])


# Set Dataset and DataLoader
train_dataset = torchvision.datasets.MNIST(root = 'c:/datasets/0811-MNIST Dataset',
                                           transform = transform,
                                           download = False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

# Set model
model = VAE()


# Set Loss Function and Optimizer
criterion = nn.BCELoss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# Train the model
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()

        recon_images, mu, logvar = model(images)

        # Calculate Reconstruction Loss
        #  - Reconstruction Loss: Differences between input images and reconstructed images
        reconstruction_loss = criterion(recon_images, images) / batch_size

        # Calculate Kullback-Leibler(KL) Divergence
        #  - Kullback-Leibler Divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Calculate Loss
        loss = reconstruction_loss + kl_divergence

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1:002d} / {num_epochs}, Loss: {loss.item():.4f}]')

torch.save(model.state_dict(), f'./outcomes/0814-mnist-vae-lr{learning_rate}.pt')




