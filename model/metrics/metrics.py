import torch
import math
import numpy as np

from deepnet.model.metrics.basemetric import Metric

class MeanAbsoluteError(Metric):
    """
    Calculates the mean absolute error.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """

    def update(self, output):
        y_pred, y = output
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._sum_of_errors += torch.sum(absolute_errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(f"{error_metric} must have at least one example before it can be computed.")
        return round(self._sum_of_errors / self._num_examples, 5)



class RootMeanSquaredError(Metric):
    
    """
    Calculates the mean squared error.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """
    def update(self, output):
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        self._sum_of_errors += torch.sum(squared_errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("RootMeanSquaredError must have at least one example before it can be computed.")
        return round(math.sqrt(self._sum_of_errors / self._num_examples), 5)



class MeanAbsoluteRelativeError(Metric):
    """
    Calculate Mean Absolute Relative Error
    """
    def update(self, output):
        y_pred, y = output
        epsilon = 0.5
        absolute_error = torch.abs(y_pred - y.view_as(y_pred)) / (torch.abs(y.view_as(y_pred)) + epsilon)
        self._sum_of_errors += torch.sum(absolute_error).item()
        self._num_examples += y.size()[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanAbsoluteRelativeError must have at least'
                                     'one sample before it can be computed.')
        return round(self._sum_of_errors / self._num_examples, 5)


def iou(output):
    y_pred, y = output
    y_pred, y = y_pred.squeeze(1), y.squeeze(1)
    intersection = (y_pred * y).sum(2).sum(1)
    union = (y_pred + y).sum(2).sum(1) - intersection

    epsilon = 1e-6
    iou = (intersection + epsilon) / (union + epsilon)
    return round( iou.sum().item()/y.shape[0], 5)
        