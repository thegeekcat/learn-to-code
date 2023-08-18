# Import modules
import torch
import torch.nn as nn


# Set parameters
latent_dim = 20


# Define a class
class VAE(nn.Module):
    # Initialize the class
    def __init__(self):
        super(VAE, self).__init__()

        # Set encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)
        )

        # Set decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    # Reparameterize latent variable
    #  - mu: Latent variable
    #  - layer: Log variance value from VAE encoder
    #          * Reason of using Log variance: To get results in positive number
    def reparameterize(self, mu, layer):
        # Set parameters
        std = torch.exp(0.5 * layer)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std


    # Define forward
    def forward(self, x):
        # Change dimension from 2D(Input) to 1D 784
        x = x.view(-1, 784)

        # Run encoder to get result of 'mean of latent variable (mu)' and 'log variance' in tensor type
        mu_logvar = self.encoder(x)

        # Get mean and log variance
        mu = mu_logvar[:, :latent_dim]
        logvar = mu_logvar[:, latent_dim:]

        # Get latent variables by sampling mu and log variance from a normal distribution
        z = self.reparameterize(mu, logvar)

        # Get reconstructed images from latent variable z
        x_recon = self.decoder(z)

        return x_recon, mu, logvar