from tqdm import tqdm

class Train:
    def __init__(self):
        return

    def fetch_data(self, batch, device):
        data, target = batch[0].to(device), batch[0].to(device)
        return data, target

    def train(self, model, train_loader, device, optimizer, criterion, losses, accuracies, scheduler):
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
        for batch_idx, batch in enumerate(pbar, 0):
            # Get samples
            data, target = self.fetch_data(batch, device)

            # Set gradients to zero before starting backpropagation
            optimizer.zero_grad()

            # Predict output
            y_pred = model(data)

            # Calculate loss
            loss = criterion(y_pred, target)

            # Perform backpropagation
            loss.backward()
            optimizer.step()
            if 'OneCycleLR' in scheduler:
                scheduler['OneCycleLR'].step()

            # Update progress bar
            pred = y_pred.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            processed += len(data)
            pbar.set_description(desc=f'Loss={loss.item():0.2f} Batch ID={batch_idx} Accuracy={(100 * correct / processed):.2f}')

        losses.append(loss)
        accuracies.append(100. * correct / processed)

    def test(self, model, val_loader, device, criterion, losses, accuracies):
        """Tests the images and prints the loss and accuracy of the model on test dataset
        Argumens:
            model: Network on which validation data predicts output
            val_loader : Dataloader which allows validation data to be iterable
            device : Machine on which model runs [gpu/cpu]
            losscriterion_function : Loss function
            losses (list): Store loss
            accuracies (list): Store accuracy
        Returns:
            Validation loss
        """

        model.eval()
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                img_batch = data  # This is done to keep data in CPU
                data, target = self.fetch_data(batch, device)  # Get samples
                output = model(data)  # Get trained model output
                val_loss += criterion(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=False)  # Get the index of the max log-probability

                correct += pred.eq(target).sum().item()
                result = pred.eq(target)

        val_loss /= len(val_loader.dataset)
        losses.append(val_loss)
        accuracies.append(100. * correct / len(val_loader.dataset))
        print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracies[-1]:.2f}%)\n')
        return val_loss, accuracies[-1]