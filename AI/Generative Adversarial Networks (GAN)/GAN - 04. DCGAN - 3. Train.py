# Import modules
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from ref_0816_cat_dcgan_model import BaseDcganGenerator, BaseDcganDiscriminator
from ref_0816_cat_dcgan_config import *

# Define a class to initialize model functions
def weights_init(m):
    classname = m.__class__.__name__

    # Normalization for Convolutional Layers
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        
    # Normalization for Batch Normalization Layers
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, val=0)



# Define a class
def main():

    # Set augmentations
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        #transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # Set dataset and DataLoader
    dataset = datasets.ImageFolder(root = data_root, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    # Visualize the augmentation process
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Training Data Image View')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                             padding=2,
                                             normalize=True).cpu(), (1, 2, 0)))
    #plt.show()
    #exit()

    # Save images
    output_images_path = './outcomes/0816-Cat_DCGAN_generated_images/'
    os.makedirs(output_images_path, exist_ok=True)


    ## ------- config
    # Set Loss Function
    criterion = nn.BCELoss()

    # Set input noise
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)


    # Load Generator model
    net_generator = BaseDcganGenerator().to(device) # DCGAN -> mean=0, stdev=0.02
    net_generator.apply(weights_init)
    #print(net_generator)


    # Load Discriminator model
    net_discriminator = BaseDcganDiscriminator().to(device)
    net_discriminator.apply(weights_init)
    #print(net_discriminator)



    # Set labels for Discriminator
    real_label = 1.
    fake_label = 0.

    # Set Optimizers
    optimizer_generator = optim.AdamW(net_generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_discriminator = optim.AdamW(net_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))


    ## ------- config

    # Initialize lists
    img_list = []
    losses_generator = []
    losses_discriminator = []
    iters = 0


    # Train model
    print('Starting training loops...')
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0) :
            ########################################
            # 1. Update Discriminating Neural Network
            #   : Maximize a result of log(D(x)) + log(1 - D(G(z)))

            # Reset gradients to 'zero'
            #  : To calculate new gradients in each batch
            net_discriminator.zero_grad()

            # Set a device
            real_gpu = data[0].to(device)

            # Set a batch size from the device
            b_size = real_gpu.size(0) # Current size: (128, 3, 64,64)

            # Set real labels for real data
            #  : Guide model to recognize samples as real
            label = torch.full((b_size,),   # Batch size in tensor format
                               real_label,  # Contents to be filled
                               dtype=torch.float,
                               device=device)

            # Get a result of evaluating the discriminator's predictions on real images
            output = net_discriminator(real_gpu).view(-1) # 'view(-1)': Flat tensor into a 1D tensor

            loss_discriminator = criterion(output, label) # # 판별자 출력 과 실제 데이터 간의 라벨 차이 계산
            loss_discriminator.backward() # 역잔파 해서 가중치 업데이트 -> 잘 구별할 수 있도록 학습 유도

            D_x = output.mean().item()

            # fack images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = net_generator(noise)
            label.fill_(fake_label)

            # D를 이용해 데이터의 진위를 판별합니다
            output = net_discriminator(fake_images.detach()).view(-1)

            # D loss
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()

            D_G_z1 = output.mean().item()

            err_discriminator = loss_discriminator + loss_d_fake
            optimizer_discriminator.step()

            ############################
            # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
            ###########################
            net_generator.zero_grad()
            label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
            # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
            # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
            output = net_discriminator(fake_images).view(-1)

            # G loss
            loss_generator = criterion(output, label)

            loss_generator.backward()
            D_G_z2 = output.mean().item()

            optimizer_generator.step()


            # 훈련 상태를 출력합니다
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(data_loader),
                         err_discriminator.item(), loss_generator.item(), D_x, D_G_z1, D_G_z2))

            losses_generator.append(loss_generator.item())
            losses_discriminator.append(err_discriminator.item())

            # fixed noise -> 6 images append
            if (iters % 500 == 0 ) or ((epoch == num_epochs -1) and (i == len(data_loader)-1)) :
                with torch.no_grad() :
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # model save
            if epoch % 5 == 0 :
                os.makedirs("./outcomes/0816-Cat_DCGAN_model_weight/", exist_ok=True)
                torch.save(netG.state_dict(), f"./outcomes/0816-Cat_DCGAN_weight/netG_epoch_{epoch}.pth")
                torch.save(netD.state_dict(), f"./outcomes/0816-Cat_DCGAN_weight/netD_epoch_{epoch}.pth")

            # epoch 5
            if (epoch + 1) % 5 == 0:
                vutils.save_image(img_list[-1], f"{output_images_path}/fake_image_epoch_{epoch}.png")

            iters +=1


if __name__ == '__main__':
    main()








