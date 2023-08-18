# Import modules
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from ref_0814_generator_model import Generator, Discriminator

# Set parameters
learning_rate = 0.0001
batch_size = 128
epochs = 400

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Call models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Set Loss function and Optimizer
criterion = nn.BCELoss()
optimizer_generator = optim.AdamW(generator.parameters(), lr=learning_rate)  # Reason of using the same parameters:
optimizer_discriminator = optim.AdamW(discriminator.parameters(), lr=learning_rate)

# Set augmentations
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])


# Set Dataloader and Dataset
dataset = torchvision.datasets.MNIST(root = 'c:/datasets/0811-MNIST Dataset',
                                     train = True,
                                     transform = transforms,
                                     download = False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Train the model
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Initialize gradient to zero
        #  : to remove existing values before calculating new gradient values in backpropagation step
        discriminator.zero_grad()

        # Get batch size
        #   : 64, 1, 28, 28
        batch_size = real_images.size(0)

        # Get labels
        real_labels = torch.ones(batch_size, 1).to(device)  # 실제 이미지에 대한 레이블을 1로 설정하여 실제 이미지를 판별할 때 사용할 레이블 생성
        fake_labels = torch.zeros(batch_size, 1).to(device) # 생성된 가짜 이미지에 대한 레이블을 0으로 설정하여, 가짜 이미지를 판별할 때 사용할 레이블 생성
        real_images = real_images.view(-1, 784).to(device)

        """
        실제 이미지를 데이터 모델에 입력하기 위해 형태 조절
          -> -1: 해당 차원의 크기 유지하면서, 다른 차원으로 크기 조정
             : 2차원 -> 1차원 백터로 변환
        """

        # Get outputs and loss for real images
        real_outputs = discriminator(real_images)  # 실제 이미지 -> 판별자 모델 -> 판별 결과
        loss_real = criterion(real_outputs, real_labels) # 실제 이미지 -> 판별자의 출력 결과 -> 실제 이미지에 대한 레이블 간의 손실 계산
                                                         # (판별자가 얼마나 실제 이미지를 정확하게 판단하는지를 나타내는 값)

        # Get outputs and loss for fake images
        noise = torch.randn(batch_size, 100).to(device)  # 생성자에 입력으로 들어갈 노이즈 백터 생성 -> 노이즈 백터는 가짜 이미지 생성에 사용 -> 100: 노이즈 백터의 차원
        fake_images = generator(noise)  # 노이즈 백터 -> 생성 모델 -> 가짜 이미지 생성
        fake_images = fake_images.view(-1, 784)  # 생성된 가짜 이미지 데이터를 판별기 모델에 넣기 위해, 1차원 벡터로 변환 필요
        fake_outputs = discriminator(fake_images.detach()) # 생성된 가짜 이미지 -> 판별 모델 -> 판별 결과
        loss_fake = criterion(fake_outputs, fake_labels) # 가짜 이미지에 대한 판별자 모델 -> 출력 결과 -> 가짜 이미지에 대한 레이블 간의 손실 계산
                                                         # (이 손실은 결국 판별자가 가짜 이미지를 얼마나 정확하게 판별하는지를 나타냄)

        # Calculate loss
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Train generator
        generator.zero_grad()  # Initialize the gradient to zero
                               # Reason of updating generator's gradient only: aim of generator is to cheat discriminator
        fake_outputs = discriminator(fake_images)
        loss_generator = criterion(fake_outputs, real_labels)  # 생성된 가짜 이미지에 대한 판별자의 출력 결과와 실제 이미지에 대한 레이블간 손실
        loss_generator.backward()
        optimizer_generator.step()


        # if i % 100 == 0:
        print(f'Epoch [{epoch+1:002d}/{epochs}], Step: [{i+1}/{len(dataloader)}], Generator Loss: {loss_generator.item():.4f}, Discriminator Loss: {loss_discriminator.item():.4f}')


    # Save images in every 5 epoch
    if (epoch + 1) % 5 == 0:
        noise = torch.randn(1, 100).to(device)
        fake_image = generator(noise)
        fake_image = 0.5 * (fake_image + 1)
        fake_image = fake_image.cpu()
        save_image(fake_image.view(1, 1, 28, 28), f'./outcomes/0814-Mnist Dataset_generated_images/4_layers/generated_image_lr{learning_rate}_epoch{epoch+1}.png', normalize=True)


# Save files
torch.save(generator.state_dict(), f'./outcomes/0814-mnist-generator-4layers-lr{learning_rate}.pth')
torch.save(generator.state_dict(), f'./outcomes/0814-mnist-discriminator-4layers-lr{learning_rate}.pth')