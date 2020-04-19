import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR

from deepnet.model.train import train
from deepnet.model.test import test

class Model:
    def __init__(
        self, model, train_loader, test_loader, device, optimizer, criterion,
        epochs, sample_count, scheduler=None
    ):
        """ Trains and validate the model
        Arguments:
            model: Model to trained and validated
            train_loader: Dataloader containing train data on the GPU/ CPU
            test_loader: Dataloader containing test data on the GPU/ CPU 
            device: Device on which model will be trained (GPU/CPU)
            optimizer: optimizer for the model
            criterion: Loss function
            epochs: Number of epochs to train the model
            sample_count: Number of data samples to be stored from validation output of correct and incorrect predictions
            scheduler: Scheduler to be applied on the model
                    (default : None)
        """
        
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.device=device
        self.optimizer=optimizer
        self.criterion=criterion
        self.epochs=epochs
        self.scheduler=scheduler
        self.sample_count=sample_count

    def fit(self):
        """Returns the training and validation results"""

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.correct_samples = []
        self.incorrect_samples= []
        last_epoch = False
        scheduler_type = self.initialize_callback(self.scheduler)

        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}:')
            if epoch == self.epochs:
                last_epoch = True
            train(
                self.model, self.train_loader, self.device, self.optimizer, self.criterion, self.train_losses,
                self.train_accuracies, scheduler_type, self.scheduler
            )
            if scheduler_type == 'StepLR':
                self.scheduler.step()
            val_loss = test(
                self.model, self.test_loader, self.device, self.criterion, self.test_losses,
                self.test_accuracies, self.correct_samples, self.incorrect_samples, last_epoch,
                self.sample_count
            )
            if scheduler_type == 'ReduceLROnPlateau':
                self.scheduler.step(val_loss)
                
        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies, self.correct_samples, self.incorrect_samples

    def initialize_callback(self, scheduler):
        """Returns the type of the scheduler given to the model"""

        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            return 'StepLR'
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return 'ReduceLROnPlateau'
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            return 'OneCycleLR'