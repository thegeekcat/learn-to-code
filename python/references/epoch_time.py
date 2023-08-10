# Import modules
import time


# Add Epoch Time in 'Train.py'



# Load models
for epoch in range(num_epochs):
    # Set a point of start time
    start_time = time.time()

    for i, (image, _) in enumerate(train_loader):
        noisy_images = add_noise(image, noise_factor=0.3)
        image = image.to('cuda')
        noisy_images = noisy_images.to('cuda')

        optimizer.zero_grad()
        outputs = model(noisy_images)

        loss = criterion(outputs.view(-1, 784),
                         image.view(-1, 784))
        loss.backward()
        optimizer.step()

    # Set a point of end time
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'Epoch: [{epoch + 1:02d} / {num_epochs}], Loss: {loss.item():.4f}, Duration: {epoch_time:.2f} secs')


# Save models as a file
torch.save(model.state_dict(), './outcomes/0811-mnist-denoising-autoencoder_lr0.001.pt')

