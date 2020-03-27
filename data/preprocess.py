from torchvision import transforms

def transformations(mean, std, horizontal_flip = 0, vertical_flip = 0, rotation = 0, random_erasing = 0, transform_train = False):
    """Transformations to be applied on the data

    Arguments:
        mean : Tuple of mean values for each channel
        std : Tuple of standard deviation values for each channel
        horizontal_flip : Probability of image being flipped horizontaly (By default : 0)
        vertical_flip : Probability of image being flipped vertically (By default : 0)
        rotation : Probability of image being rotat (By default : 0)
        random_erasing : Probability of image being flipped horizontaly (By default : 0)
        transform_train : If True, transformations for training data else for testing data (By default : False)
        
    Returns:
        Transformations that is to applied on the data
    """
    
    transformations = []
    
    if transform_train:
        transformations.append(transforms.RandomHorizontalFlip(horizontal_flip))
        transformations.append(transforms.RandomVerticalFlip(vertical_flip))
        transformations.append(transforms.RandomRotation(rotation))

    transformations.extend([transforms.ToTensor(),
    transforms.Normalize(mean, std)])

    if transform_train:
        transformations.append(transforms.RandomErasing(random_erasing))

    return transforms.Compose(transformations)




