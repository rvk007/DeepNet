# DeepNet

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g63kM2rq3pktpTx5neqNlbSVYeT9xvEk)  

Deepnet is an open-source library that can be used for solving problems of Computer vision in Deep Learning.  
 
NOTE: This documentation applies to the MASTER version of DeepNet only.


## Install Dependencies

Install the required packages  
`pip install -r requirements.txt`

## Features

DeepNet currently supports the following features:

### Models

| Models | Description |
| ------- | ----- |
| [ResNet](./model/models/resnet.py) | ResNet-18 |
| [ResModNet](./model/models/resmodnet.py) | A modified version of ResNet-18  |
| [CustomNet](./model/models/customnet.py) | A modified version of ResNet-18   |
| [MaskNet3](./model/models/masknet.py) | A model to predict the Segmentation mask of the given image. |
| [DepthMaskNet8](./model/models/depthnet.py) | A model to predict the Monocular Depth Maps of the given image. |

### Loss

| Models | Description |
| ------- | ----- |
| [Dice](./model/losses/dice_loss.py) | ResNet-18 |
| [SSIM](./model/losses/ssim.py) | A modified version of ResNet-18  |
| [MSE]() | Mean squared error (squared L2 norm) between each element in the input and target   |
| [BCE]() | Binary Cross Entropy
    between the target and the output |
| [BCEWithLogitsLoss]() | Combination of Sigmoid layer and the BCE in one single
    class |
| [RMSE]() | Root mean squared error (squared L2 norm) between each element in the input and target |

Weighted Combination of loss functions
- [BCE-RMSE]()
- [SSIM-RMSE]()
- [BCE-SSIM]()
- [RMSE-SSIM]()
- [SSIM-DICE]()
- [RMSE-DICE]()
- [BCE-Dice]()
- [Rmse-Bce-Dice]()

### Utilities

| Models | Description |
| ------- | ----- |
| [GRADCAM](./gradcam/gradcam.py) | Calculates GradCAM saliency map |
| [GradCAMpp](./gradcam/gradcam_pp.py) | Calculate GradCAM++ salinecy map using heatmap and image |
| [LRFinder](./lr_finder/lr_finder.py) | Range test to calculate optimal Learning Rate  |
| [Checkpoint](./utils/checkpoint.py) | Loading and saving checkpoints  |
| [ProgressBar](./utils/progress_bar.py) | Display Progress bar |
| [Tensorboard](./utils/tensorboard.py) | Creates Tensorboard visualization  |
| [Summary](./utils/summary.py)| Display model summary |
| [Plot](./utils/plot.py)| Plot the graph of a metric, prediction image and class accuracy |



### Training and Validation

| Models | Description |
| ------- | ----- |
| [Train](/home/rvk/DeepNet/model/train.py) | Training and Validation of the model |
| [Model](/home/rvk/DeepNet/model/learner.py) | Handles all the function for training a model  |
| [Dataset](/home/rvk/DeepNet/data/dataset) | Contains classes to handle |


### Metrics

- [Mean Absolute Error](https://github.com/rvk007/DeepNet/blob/f67732d2d65798289925ea76d58f1d8636f13273/model/metrics.py#L36)
- [Root Mean Squared Error](https://github.com/rvk007/DeepNet/blob/f67732d2d65798289925ea76d58f1d8636f13273/model/metrics.py#L50)
- [Mean Absolute Relative Error](https://github.com/rvk007/DeepNet/blob/f67732d2d65798289925ea76d58f1d8636f13273/model/metrics.py#L67)
- [Intersection Over Union Error](https://github.com/rvk007/DeepNet/blob/f67732d2d65798289925ea76d58f1d8636f13273/model/metrics.py#L84)
- [Root Mean Square Error](https://github.com/rvk007/DeepNet/blob/f67732d2d65798289925ea76d58f1d8636f13273/model/metrics.py#L130)

### Scheduler
- StepLR
- ReduceLROnPlateau
- OneCycleLR

For a demo on how to use these modules, refer to the notebooks present in the [examples](./examples) directory.

## Contact/Getting Help

If you need any help or want to report a bug, raise an issue in the repo.