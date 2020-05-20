import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR

from deepnet.utils.checkpoint import Checkpoint

class Model:
    def __init__(self, model, dataset, train_loader, test_loader, device, optimizer, criterion, 
        epochs, metrics, callbacks=None
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
            callbacks: Callbacks to be applied on the model
                    (default : None)
        """
        
        self.model=model
        self.dataset = dataset
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.device=device
        self.optimizer=optimizer
        self.criterion=criterion
        self.epochs=epochs
        self.metrics = metrics
        self.callbacks=callbacks

    def fit(self):
        """Returns the training and validation results"""

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.last_saved_epoch = 0
        
        self.scheduler = {}
        self.checkpoint = None
        self.initialize_callbacks()

        if self.checkpoint:
            self.last_metric, self.mode = self.checkpoint.checkpoint_metric()
            self.verbose = self.checkpoint.verbose


        for epoch in range(1, self.epochs + 1):
            if self.checkpoint.last_reload:
                self.reload_last_checkpoint()

            print(f'Epoch {epoch}:')

            self.dataset.train(
                self.model, self.train_loader, self.device, self.optimizer, self.criterion, self.metrics, self.train_losses,
                self.train_accuracies, self.scheduler
            )

            if 'StepLR' in self.scheduler:
                self.scheduler['StepLR'].step()

            val_loss, val_acc = self.dataset.test(
                self.model, self.test_loader, self.device, self.criterion, self.metrics, self.test_losses,
                self.test_accuracies,last_epoch
            )

            if 'ReduceLROnPlateau' in self.scheduler:
                self.scheduler['ReduceLROnPlateau'].step(val_loss)

            if self.ischeckpoint(epoch, val_acc, val_loss):
                state = {
                    'epoch' : self.last_saved_epoch + epoch + 1,
                    'accuracy' : val_acc,
                    'loss' : val_loss,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'scheduler_state_dict' : self.scheduler.state_dict(),
                }

                self.checkpoint.save(state)
                
        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies
    def initialize_callbacks(self):
        """Returns the type of the scheduler given to the model"""

        for callback in self.callbacks:
            if isinstance(callback, torch.optim.lr_scheduler.StepLR):
                self.scheduler['StepLR'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler['ReduceLROnPlateau'] = callback
            elif isinstance(callback, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler['OneCycleLR'] = callback
            elif isinstance(callback, Checkpoint):
                self.checkpoint = callback
    
    def ischeckpoint(self, epoch, accuracy, loss):
        monitor = self.checkpoint.monitor

        if self.checkpoint.save_best_only:
            if self.mode == 'max':
                if accuracy > self.last_metric:
                    self.last_metric = accuracy
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, accuracy))
                    return True
            elif self.mode == 'min':
                if loss < self.last_metric:
                    self.last_metric = loss
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, loss))
                    return True
            if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, monitor, self.last_metric))
            return False
        else:
            if self.verbose > 0:
                if monitor == 'Accuracy' and accuracy > self.last_metric:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, accuracy))
                    self.last_metric = accuracy

                elif monitor == 'Loss' and loss < self.last_metric:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, monitor, self.last_metric, loss))
                    self.last_metric = loss
                    
                else:
                    print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, monitor, self.last_metric))

            return True

    def reload_last_checkpoint(self):

        data = self.checkpoint.last_checkpoint

        self.last_saved_epoch = data['epoch']
        self.last_accuracy = data['accuracy']
        self.last_loss = data['loss']
        self.model.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
