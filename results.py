import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def display(val_data,device,model,classes):
    """
    Displays the incorrect prediction 

    Arguments:
        val_data : Validation data of the dataset
        device : Machine on which model runs [gpu/cpu]
        model : Network on which data learns
        classes : Target Classes of the dataset
    """
    # Set number of images to display
    num_images = 4
    images, targets = val_data.data, val_data.targets
    images_idx = random.sample(range(0, images.shape[0]), num_images)

    # Define classes of CIFAR dataset

    fig, axs = plt.subplots(1, num_images, figsize=(8, 8))
    fig.tight_layout()

    with torch.no_grad():
        count = 0
        for idx in images_idx:
            # Prepare images and targets
            image = images[idx]
            target = targets[idx]
            images_tensor = torch.Tensor(
                np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)
            )

            # Get model predictions
            images_cuda = images_tensor.to(device)
            outputs = model(images_cuda)
            _, predicted = torch.max(outputs, 1)

            axs[count].axis('off')
            axs[count].set_title(f'Label: {classes[target]}\nPrediction: {classes[predicted.item()]}')
            axs[count].imshow(image)
            count += 1