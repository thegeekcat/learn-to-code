# Import modules
import torch.nn as nn

# Set a parameter for latent variable
latent_dim = 20

# Define a class
class DenoisingAutoEncoder(nn.Module):

    # Initialize the class
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()

        # Set the Encoder
        self.encoder = nn.Sequential(
            # Size of input images: 32 x 32 -> 784
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )

        # Set the Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()
        )

    # Define a forward function
    def forward(self, x):
        # Reshape the model
        #  : Reshaping the input tensor 'x' to have a shape of (-1, 784)
        # view:
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded



