# Import modules
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from ref_0817_style_transfer_loss import ContentLoss, StyleLoss, gram_matrix

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set output size
img_size = 512 if torch.cuda.is_available() else 128  # '512': when GPU is available, '128': when GPU is not available


# Set augmentations
#  Note: Usually less use augmentations -> usually use resize function
loader_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])


# Define a class for the image loader
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader_transforms(image).unsqueeze(0)
    return image.to(device, torch.float)


# Set images
style_img = image_loader('g:/My Drive/datasets/0816-style-transfer-whale01.jpg')
content_img = image_loader('g:/My Drive/datasets/0816-style-transfer-cat01.jpg')

# Change to PIL again
unloader = transforms.ToPILImage()


# Define a class to display images
def imshow(tensor, title=None):
    # Copy tensors
    image = tensor.cpu().clone()

    # Remove added dimension
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # Pause while updating plots


# Display images
plt.figure()
imshow(style_img, title='Style Image')
plt.figure()
imshow(content_img, title='Content Image')
plt.show()

# Set the model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Calculate mean and std
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Define a class
class Normalization(nn.Module):
    # Initialize the class
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        # self.mean = torch.tensor(mean).view(-1, 1, 1).clone().detach()
        # self.std = torch.tensor(std).view(-1, 1, 1).clone().detach()
        #print('self.mean, self.std: ', self.mean, self.std)

    # Define forward
    def forward(self, img):
        # Return normalized image
        return (img - self.mean) / self.std


# Set layers
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



# Define a function for model and losses
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers_default, style_layers=style_layers_default):
    """
    :param cnn: Convolutional neural network VGG Model
    :param normalization_mean:
    :param normalization_std:
    :param style_IMG:
    :param content_img:
    :param content_layers: Layers for content
    :param style_layers: Layers for style
    :return:
    """
    #print('style image: ', style_img)
    #print('content image: ', content_img)

    # Set normalization
    #  : Nomarlization using Mean and Std
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    #print('normalization: ', normalization)
    #print('normalization mean, normalization std: ', normalization_mean, normalization_std)

    # Set lists of losses
    content_losses = []
    style_losses = []

    # Set a model
    #  : Set a starting point for neural network by initializing layers only containing normalized layer
    #
    model = nn.Sequential(normalization)

    # Add layers for Style Transfer
    #  - To describe the intermediate steps in model creation for Style Transfer
    #  - Check CNN model's each layer and add layers based on its types
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}' # Rename as 'conv_x'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}' # Rename as 'relu_x'
            layer = nn.ReLU()  # Create a new layer and make it 'nn.ReLU()''
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}' # Rename as 'pool_x'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'   # Rename as 'bn_x'
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        # Add newly added names and layers to model
        model.add_module(name, layer)

        # Calculate Content Loss and add it to model when 'name' is in the Content Layer
        if name in content_layers:
            # Add Content Loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target) # Calculate Content Loss for 'target'
            model.add_module('content_loss_{}'.format(i), content_loss) # Add Content Loss to model
            content_losses.append(content_loss)

        # Add
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)



    # 손실 레이어 이후의 레이어들은 주로 스타일 전이(style transfer)에서 손실을 계산하는 역할을 수행하며,
    # 이전 레이어들은 이미지에 대한 변형을 담당하거나 중간 특성을 추출하는 역할
    for i in range(len(model) -1, -1, -1):
        # Check the current layer
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    # Modify model structure
    #  : Remove unnecessary layers by keeping only set layers (i+1)
    model = model[ : (i + 1)]

    return model, style_losses, content_losses


# Set input images
input_img = content_img.clone()
#input_img = torch.clamp(input_img, 0, 1)


# Define a function for input optimizer
def get_input_optimizer(input_img):
    # 'LBFGS'
    #  : to solve a large scale of non-linear optimization issues with limited memory
    optimizer = optim.LBFGS([input_img])

    return optimizer


# Set a function to run Style Transfer
#  : style_weight=1000000, content_weight=1
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img,
                       num_steps=1500, style_weight=1000000, content_weight=1):
    print('Building the Style Transfer Model...')

    # Get model and losses
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)

    #
    input_img.requires_grad_(True) # Calculate backpropagation: Calculate slope for input images
    model.requires_grad_(False)  # 'requires_grad_(False)': Not update weights during training in VGG models

    # Set optimizer
    #  : Optimize input images to generate images that are most suitable for styles and contents
    optimizer = get_input_optimizer(input_img)

    print('Optimizing....')

    # Run the model
    run = [0]
    while run[0] <= num_steps:
        # Define a closure function
        def closure():
            # Make sure image pixels are within correct range
            with torch.no_grad():
                input_img.clamp_(0, 1)  # 'clamp_(0, 1)': Clamp between 0 and 1

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            # Set Losses
            for s_loss in style_losses:
                style_score += s_loss.loss
            for c_loss in content_losses:
                content_score += c_loss.loss

            # Calculate scores
            style_score *= style_weight
            content_score *= content_weight

            # Calculate all loss
            loss = style_score + content_score
            loss.backward()

            #
            run[0] += 1
            if run[0] % 50 == 0:
                print('run {}: '.format(run))
                print('Style Loss: {:.4f}, Content Loss: {:.4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

# Set output
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img=content_img, style_img=style_img, input_img=input_img)


# Visualization
plt.figure()
imshow(output, title='Output Images')
plt.ioff()
plt.show()
















