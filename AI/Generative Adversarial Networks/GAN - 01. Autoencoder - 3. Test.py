# Import modules
import torch
import matplotlib.pyplot as plt
from ref_0810_mnist_model import Autoencoder


# Load models
load_auto_encoder = Autoencoder()
load_auto_encoder.load_state_dict(torch.load('./outcomes/0811-mnist-checkpoint_lr0.005.pt', map_location='cuda'))
load_auto_encoder.eval()

with torch.no_grad():
    test_sample = torch.rand(1, 32)  #
    #print(test_sample) # Result: tensor([[0.9190, 0.6612, 0.7758, 0.4318, 0.3522, 0.8672, 0.6371, 0.0432, 0.3760,
    generated_sample = load_auto_encoder.decoder(test_sample).view(1, 1, 28, 28)
                        # 'load_aut'
                        # 'view(1, 1, 28, 28)': Batch_size, ?, size, size

plt.imshow(generated_sample.squeeze().numpy(), cmap='gray')
plt.show()