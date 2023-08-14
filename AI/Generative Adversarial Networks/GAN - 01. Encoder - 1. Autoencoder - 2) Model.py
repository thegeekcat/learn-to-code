# Import modules
import torch.nn as nn


# Define a class
class Autoencoder(nn.Module):

    """
    Size of input images: 28 * 28 -> 784

    """

    # Initialize the class
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Set backbones for encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Set backbones for decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    # Set a forward
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
