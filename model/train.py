from tqdm import tqdm

def train(model, train_loader, device, optimizer, criterion, losses, accuracies, scheduler_type, scheduler):
    """Trains the images and prints the loss and accuracy of the model on train dataset
    Arguments:
        model: Network on which training data learns
        val_loader : Dataloader which allows training data to be iterable
        device : Machine on which model runs [gpu/cpu]
        criterion : Loss function
        losses (list): Store loss
        accuracies (list): Store accuracy
        scheduler_type (str): Scheduler name
        scheduler: Scheduler to be applied on the model
    """
    
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar, 0):
        # Get samples
        data, target = data.to(device), target.to(device)

        # Set gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Predict output
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        # Perform backpropagation
        loss.backward()
        optimizer.step()
        if scheduler_type == 'OneCycleLR':
            scheduler.step()

        # Update progress bar
        pred = y_pred.argmax(dim=1, keepdim=False)
        correct += pred.eq(target).sum().item()
        processed += len(data)
        pbar.set_description(desc=f'Loss={loss.item():0.2f} Batch ID={batch_idx} Accuracy={(100 * correct / processed):.2f}')

    losses.append(loss)
    accuracies.append(100. * correct / processed)