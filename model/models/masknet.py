import torch
import torch.nn as nn
import torch.nn.functional as F

from deepnet.model.learner import Model

class MaskNet2(nn.Module):
    def __init__(self):
        super(MaskNet2, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.last= nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        v = x['bg']
        z = x['bg_fg']
        # v=z=x
        v = self.conv_layer(v)
        z = self.conv_layer(z)

        xx = torch.cat([v,z], dim=1)
        xx = F.interpolate(xx,  scale_factor=2, mode='bilinear', align_corners=True)
        xx = self.last(xx)

        return xx

    def learner(self, model, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks):
        """Trains the model
        Arguments:
            model: Model to trained and validated
            train_loader: Dataloader containing train data on the GPU/ CPU
            test_loader: Dataloader containing test data on the GPU/ CPU 
            device: Device on which model will be trained (GPU/CPU)
            optimizer: optimizer for the model
            criterion: Loss function
            epochs: Number of epochs to train the model
            callbacks: Scheduler to be applied on the model
                    (default : None)
        """

        learn = Model(model, dataset_train, train_loader, test_loader, device, optimizer, criterion, epochs, metrics, callbacks)
        self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies = learn.fit()

    @property
    def results(self):
        """Returns model results"""
        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies