# LWIR cloud image segmentation model

Segment clouds in satellite infrared single channel images

## General method 
Supervised 

Transfer-learning of VGG19 model for feature extraction and an added pixel wise classification layer for instance segmentation.


## Data set 
Pairs of IR (single channel) images and binary masks for supervised learning.
Image size: 1024x2024 

Training set: 1141 
Validation set: 32
(No extra test set)

## Model card 

## Training 

### Data preprocessing 
- TIFF images 
- Adaption to multi channel base model, by replicating gray scale input image to 3 channels  
- 

### Model training  

## Evaluation  
F1 score (pixel wise): 
Binary IOU: 

