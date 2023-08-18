# Import modules
import torch.nn as nn



# Define a class for generator
class Generator(nn.Module):
    # Initialize the class
    def __init__(self):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.Linear(100, 120),
            nn.ReLU(),
            nn.Linear(120, 360),
            nn.ReLU(),
            nn.Linear(360, 560),
            nn.ReLU(),
            nn.Linear(560, 784),
            nn.Tanh()
        )

        pass

    # Define forward
    def forward(self, x):
        return self.generator(x)


# Define a class for discriminator
class Discriminator(nn.Module):
    # Initialize the class
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(784, 560),
            nn.LeakyReLU(0.2),  # Default: 0.1 - 0.3
            nn.Linear(560, 360),
            nn.ReLU(),
            nn.Linear(360, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Reason of using Sigmoid: Get 0 or 1 values to identify '1=real' and '0=fake'
        )

    # Define forward
    def forward(self, x):
        return self.discriminator(x)