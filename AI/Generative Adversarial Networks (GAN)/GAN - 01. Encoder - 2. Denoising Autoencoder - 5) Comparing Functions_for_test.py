# Import modules
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Set a latent value
latent_dim = 20
latent_vector = torch.rand(1, latent_dim)

# Define a class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 784)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


# Call a class
decoder = Decoder()

# Set hyper parameteres for Tanh function
decoder_tanh = decoder(latent_vector)
decoder_tanh = decoder_tanh.view(28, 28).detach().numpy()

# Set hyper parameteres for ReLU function
decoder.tanh = nn.ReLU()
decoder_relu = decoder(latent_vector)
decoder_relu = decoder_relu.view(28, 28).detach().numpy

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(decoder_tanh, cmap='gray')
axes[0].set_title('Decoded with Tanh')
axes[0].axis('off')

axes[0].imshow(decoder_relu, cmap='gray')
axes[0].set_title('Decoded with ReLU')
axes[0].axis('off')

plt.imshow()
