import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

class Transformations:
    def __init__(
    self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), horizontal_flip = 0, vertical_flip = 0, rotate_degree = 0, rotation = 0, cutout = 0,
    cutout_height= 8, cutout_width = 8, gaussian_blur = 0,train=False
    ):

        """Transformations to be applied on the data
        Arguments:
            mean : Tuple of mean values for each channel
            std : Tuple of standard deviation values for each channel
            horizontal_flip : Probability of image being flipped horizontaly
                (default : 0)
            vertical_flip : Probability of image being flipped vertically
                (default : 0)
            rotate_degree : Maximum degree by which image should be rotated
            rotation : Probability of image being rotated
                (default : 0)
            cutout : Probability of image being cutout
                (default : 0)
            cutout_height: Maximum height to be cutout from the image
                (default : 8)
            cutout_width : Maximum width to be cutout from the image
                (default : 8)
            gaussian_blur : Probability of applying gaussian blur on the image
                (default : 0)
            transform_train : If True, transformations for training data else for testing data
                (default : False)
            
        Returns:
            Transformations that is to applied on the data
        """
        
        transformations = []
        if train:
            transformations.append(A.HorizontalFlip(p=horizontal_flip))
            transformations.append(A.VerticalFlip(p=vertical_flip))
            transformations.append(A.GaussianBlur(p=gaussian_blur))
            transformations.append(A.Rotate(limit=rotate_degree, p=rotation))
            transformations.append(A.CoarseDropout(max_holes=1, fill_value=tuple(x*255 for x in mean),
                                                   max_height=cutout_height,max_width=cutout_width,
                                                   min_height=1, min_width=1, p=cutout))

        transformations.append(A.Normalize(mean=mean, std=std,always_apply=True))
        transformations.append(ToTensor())

        self.transform =  A.Compose(transformations)

    def __call__(self, image):
        """
        Transform the image through the data transformation pipeline

        Arguments:
            image : Image to be transformed

        Returns:
            Transformed image
        """
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image