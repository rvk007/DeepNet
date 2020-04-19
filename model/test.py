import torch

def test(model, val_loader, device, criterion, losses, accuracies, correct_samples, 
         incorrect_samples, last_epoch, sample_count):
    """Tests the images and prints the loss and accuracy of the model on test dataset
    Argumens:
        model: Network on which validation data predicts output
        val_loader : Dataloader which allows validation data to be iterable
        device : Machine on which model runs [gpu/cpu]
        losscriterion_function : Loss function
        losses (list): Store loss
        accuracies (list): Store accuracy
        correct_samples (list): Store image, label, prediction of the correct output
        incorrect_samples (list): Store image, label, prediction of the incorrect output 
        last_epoch (bool): True if in last epoch else False
        sample_count: Number of data samples to be stored from validation output of correct and incorrect predictions
    Returns:
        Validation loss
    """

    model.eval()
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            img_batch = data  # This is done to keep data in CPU
            data, target = data.to(device), target.to(device)  # Get samples
            output = model(data)  # Get trained model output
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # Get the index of the max log-probability

            correct += pred.eq(target).sum().item()
            result = pred.eq(target)

            if last_epoch:
                for i in range(len(result)):
                    if result[i] and len(correct_samples)<sample_count:
                        correct_samples.append({'image' : img_batch[i],
                                                'label' : list(target.view_as(pred))[i],
                                                'prediction' : list(pred)[i]}
                                              )
                    elif not result[i] and len(incorrect_samples)<sample_count:
                        incorrect_samples.append({'image' : img_batch[i],
                                                'label' : list(target.view_as(pred))[i],
                                                'prediction' : list(pred)[i]}
                                                )

    val_loss /= len(val_loader.dataset)
    losses.append(val_loss)
    accuracies.append(100. * correct / len(val_loader.dataset))
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracies[-1]:.2f}%)\n')
    return val_loss