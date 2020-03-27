import torch

def test(model, val_loader, device, loss_function, losses, accuracies):
    """
    Tests the images and prints the loss and accuracy of the model on test dataset

    Argumens:
        model: Network on which validation data predicts output
        val_loader : Dataloader which allows validation data to be iterable
        device : Machine on which model runs [gpu/cpu]
        loss_function : Loss function
        losses : List to store and return loss
        accuracies : List to store and return accuracy
    """

    model.eval()
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            img_batch = data  # This is done to keep data in CPU
            data, target = data.to(device), target.to(device)  # Get samples
            output = model(data)  # Get trained model output
            val_loss += loss_function(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # Get the index of the max log-probability

            correct += pred.eq(target).sum().item()
    
    val_loss /= len(val_loader.dataset)
    losses.append(val_loss)
    accuracies.append(100. * correct / len(val_loader.dataset))
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracies[-1]:.2f}%)\n')