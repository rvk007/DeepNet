# Evaluation Metrics

In object detection, evaluation is non trivial, because there are two distinct tasks to measure:
- Determining whether an object exists in the image (classification)
- Determining the location of the object (localization, a regression task).

So we need other mertrics to evaluate the performance of our model.

## Results

| Precision   |   Recall      | F1 Score    |      IoU      |      mAP      |
| :---------: | :-----------: | :---------: | :-----------: |:------------: |
| 0.75        | 0.586         |    0.658    | 0.512         |   0.662       |

## Precision
Precision quantifies the number of positive class predictions that actually belong to the positive class. 
![precision](images/precision.png)

Precision talks about how precise/accurate our model is, out of those predicted positive, how many of them are actual positive. It is a good measure to determine, when the costs of False Positive is high.  
 Note: If you have a precision score of close to 1.0 then there is a high likelihood that whatever the classifier predicts as a positive detection is in fact a correct prediction.

## Recall
Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.
![recall](images/recall.png)

 Recall actually calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Applying the same understanding, we know that Recall shall be the model metric we use to select our best model when there is a high cost associated with False Negative.  
 Note: If you have a recall score close to 1.0 then almost all objects that are in your dataset will be positively detected by the model.

## F1 Score
 F1 which is a function of Precision and Recall. This provides a single score that balances both the concerns of precision and recall in one number.
 ![f1score](images/f1score.png)

 F1 Score might be a better measure to use if we need to seek a balance between Precision and Recall.


## IoU
Intersection over Union is a ratio between the intersection and the union of the predicted boxes and the ground truth boxes.
It judges the correctness of each of these detections predicted by the model.
We use a threshold IoU value, that means if the IoU of predicted output is greater than the threshold value only then it is considered as True positive else it becomes False Positive.  
 ![iou](images/iou.png)

 **Confidence score** is the probability that an anchor box contains an object. It is usually predicted by a classifier.


## Mean Average Precision (mAP)  


Every image in an object detection problem could have different objects of different classes. Hence, the standard metric of precision cannot be directly applied here. That is why we apply Mean Average Precision( mAP). 

Using IoU and Confidence score we can calculate **Precision** and **Recall** metrics which is used for the precision-recall curve for the calculation of mAP.

Average Precision(AP) is the precision averaged across all unique recall levels for one class. The mAP is the average of the AP calculated for all the classes in the dataset. 

 ![map](images/map.png)



Note:
True Positives: Output labeled as positive that are actually positive  
False Positives: Output labeled as positive that are actually negative  
True Negatives: Output labeled as negative that are actually negative  
False Negatives: Output labeled as negative that are actually positive  
![confusion_matrix](images/confusion_matrix.png)