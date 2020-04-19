import numpy as np

from deepnet.data.cifar10_download import download
from deepnet.data.preprocess import Transformations
from deepnet.data.dataloader import data_loader

class CIFAR10:
    def __init__(
        self, cuda=False, batch_size=1, num_workers=1, pad_dim = (0,0),
        random_crop_dim = (0,0), horizontal_flip=0.0, vertical_flip=0.0,
        rotate_degree = 0.0, rotation=0.0, cutout=0.0, cutout_dim = (1,1),
        gaussian_blur=0.0
    ):
        """Initializes the CIFAR-10 dataset
        Arguments:
            cuda: True if machine is GPU else False if CPU
                (default: False)
            batch_size: Number of images in a batch
                (default: 1)
            num_workers: Number of workers simultaneously putting data into RAM
                (default: 1)
            pad_dim (list, optional): Pad side of the image
                pad_dim[0]: minimal result image height (int)
                pad_dim[1]: minimal result image width (int)
                (default: (0,0))
            random_crop_dim (list, optional): Crop a random part of the input
                random_crop_dim[0]: height of the crop (int)
                random_crop_dim[1]: width of the crop (int)
                (default: (0,0))
            horizontal_flip: Probability of image being flipped horizontaly 
                (default: 0)
            vertical_flip: Probability of image being flipped vertically 
                (default: 0)
            rotation: Probability of image being rotated 
                (default: 0)
            cutout: Probability of image being cutout 
                (default: 0)
            cutout_dim (list, optional): Cutout a random part of the image
                cutout_dim[0]: height of the cutout (int)
                cutout_dim[1]: width of the cutout (int)
                (default: (1,1))
            guassian_blur: Probability of image being blurred using guassian_blur
                (default: 0)
        """
        
        self.cuda = cuda
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pad_dim = pad_dim
        self.random_crop_dim = random_crop_dim
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.cutout = cutout
        self.cutout_dim = cutout_dim
        self.gaussian_blur = gaussian_blur

        self._classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
        )

        self._sample_data = self.download_cifar10()

        self._train_transformations = self.transform()
        self._train_data = self.download_cifar10(train = True, apply_transformations = True)

        self._test_transformations = self.transform(train = False)
        self._test_data = self.download_cifar10(train = False, apply_transformations = True)
        
    @property
    def test_data(self):
        """Returns Test Dataset"""
        return self._test_data

    @property
    def classes(self):
        """Returns Target classes of the dataset"""
        return self._classes

    @property
    def data(self):
        """Returns Features and target of dataset"""
        return self._sample_data.data, self._sample_data.targets

    @property
    def mean(self):
        """Returns Channel-wise mean of the whole dataset"""
        return tuple(np.mean(self._sample_data.data, axis = (0,1,2))/255)

    @property
    def std(self):
        """Returns Channel-wise standard deviation of the whole dataset"""
        return tuple(np.std(self._sample_data.data, axis = (0,1,2))/255)

    @property
    def input_size(self):
        """Returns Dimension of the input image"""
        _, height, width, channels = self._sample_data.data.shape
        return tuple((channels, height, width))
    
    def transform(self, train=True):
        """Creates transformations to be applied
        Arguments:
            train : True if tranformations to be applied on train dataset, False for test dataset
                (default : True)     
        Returns:
            Transformations
        """
        args = {'mean':self.mean,
                'std':self.std}

        if train:
            args['train'] = True
            args['pad_dim'] = self.pad_dim
            args['random_crop_dim'] = self.random_crop_dim
            args['horizontal_flip'] = self.horizontal_flip
            args['vertical_flip'] = self.vertical_flip
            args['rotation'] = self.rotation
            args['cutout'] = self.cutout
            args['cutout_dim'] = self.cutout_dim
            args['gaussian_blur'] = self.gaussian_blur
            
        return Transformations(** args)

    def download_cifar10(self, train=True, apply_transformations=False):
        """Download the dataset
        Arguments:
            train: True if train data is to downloaded, False for test data
                (default: True)
            apply_transformations: True if transformations is to applied, else False
                (default: False)
        Returns:
            Dataset after downloading
        """
        if apply_transformations:
            transform = self._train_transformations if train else self._test_transformations
            return download(train, transform)
        else:
            return download()

    def dataloader(self, train=False):
        """Creates the dataloader
        Arguments:
            train: True when train data is to be loaded, False for test data
                (default: False)
        Returns:
            Dataloader, after loading the data on the GPU or CPU
        """
        data = self._train_data if train else self._test_data
        return data_loader(data, self.batch_size, self.num_workers, self.cuda)