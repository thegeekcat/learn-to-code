# Import modules
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# Load Images
image_path = './data/llama-02.png'
image = Image.open(image_path)

# Convert image to a Pytorch tensor
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).float()
#print(image_tensor)

# Set size of grid
grid_size = 16
height, width = image_tensor.shape[1], image_tensor.shape[2]
grid_width = width // grid_size
grid_height = height // grid_size

# Create grid
grids = []
for i in range(grid_size):
    for j in range(grid_size):
        x_min = j * grid_width
        y_min = i * grid_height
        x_max = (j + 1) * grid_width
        y_max = (i + 1) * grid_height
        grid = image_tensor[:, y_min:y_max, x_min:x_max]
        grids.append(grid)

# Visualization
fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
for i in range(grid_size):
    for j in range(grid_size):
        axs[i, j].imshow(grids[i * grid_size + j].permute(1, 2, 0))
        axs[i, j].axis('off')
plt.show()