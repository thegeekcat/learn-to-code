# Import modules
import torch
import torchvision
import matplotlib.pyplot as plt
from ref_0811_mnist_model import DenoisingAutoEncoder

# Set augmentations
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5,))
])

# Set parameters
batch_size = 10

# Load a model
model = DenoisingAutoEncoder()
model.load_state_dict(torch.load('./outcomes/0811-mnist-denoising-autoencoder_lr0.001.pt', map_location='cpu'))
model.eval()


# Set Dataset and DataLoader
test_dataset = torchvision.datasets.MNIST(root='c:/datasets/0811-MNIST Dataset',
                                          train = False,
                                          transform = transform,
                                          download = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)

# Load images
for images, _ in test_loader:
    # Add noise
    noise_factor = 0.5
    noisy_images = images + noise_factor * torch.randn(images.size())

    # Run the model
    reconstructed_images = model(noisy_images)


    # Visualization
    for j in range(batch_size):
        # Set plots
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))

        # Set original images
        original_images = images[j].view(28, 28)
        axes[0].imshow(original_images.detach(), cmap='gray')
        axes[0].set_title('Original Images')

        # Set noisy images
        noisy_images = noisy_images[j].view(28, 28)
        axes[1].imshow(noisy_images.detach(), cmap='gray')
        axes[1].set_title('Noisy Images')

        # Set reconstructed images
        reconstructed_images = reconstructed_images[j].view(28, 28)
        axes[2].imshow(reconstructed_images.detach(), cmap='gray')
        axes[2].set_title('Reconstructed Images')

        for ax in axes:
            ax.axis('off')

        plt.show()
