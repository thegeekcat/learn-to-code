# Import modules
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


# Set a batch size
batch_size = 128

# Set transforms
transforms = transforms.Compose([
    transforms.RandAugment(),
    transforms.ToTensor(),  # Notes: 'ToTensor()' should be run after AUGMENTATION as augmentation runs with images
    transforms.Normalize((0.5, ), (0.5, ))  # 'Normalize(Mean, std)'   # Notes: 'Normalize()' should be run after 'ToTensor()' as it needs Tensor type
])


# Set a dataset
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

    # Clamp images
    #  : Constraining a value within a specific range, setting it to a minimum or maximum limit if it falls outside of that range
    noisy_images = torch.clamp(noisy_images, # Source images
                               -1,           # Minimum value  (When normalized)
                               1)            # Maximum value  (When normalized)


    return noisy_images

# Load images
for images, _ in train_loader:
    #print(images.shape)
    #exit()
    noisy_images = add_noise(images)
    break
    
# Visualization
fix, axes = plt.subplots(1, 2, figsize=(10, 5))

"""
8개 이미지선택하고 torchvision.utils.make_grid 함수를 사용하여 이미지 그리드생성 (padding : 이미지 간격, 정규화, 이미지 차원 
-> RGB 채널이 마지막 차원이 되도록 합니다.
"""
# Original Images
axes[0].imshow(np.transpose(torchvision.utils.make_grid(images[:8], padding = 2,  normalize = True),
                            (1, 2, 0)))
axes[0].set_title('Original Images')
axes[0].axis('off')

# Noisy Images
axes[1].imshow(np.transpose(torchvision.utils.make_grid(noisy_images[:8], padding = 2,  normalize = True),
                            (1, 2, 0)))
axes[1].set_title('Original Images')
axes[1].axis('off')

plt.show()


