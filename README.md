# LWIR cloud image segmentation 

Segment clouds in satellite infrared single channel images.

## Approach 
Supervised training of a model to learn the segmentation mask as a target labels (Y) from the satellite images (X).

Two base architectures for transfer-learning of pretraine models: 

1) Convolutional VGG19 base model for feature extraction and a pixel wise classification layer on top. 
2) ViT architecture based Segformer base model. 

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
- Augmentations: 

### Model training  


## Evaluation  
F1 score: 
Binary IOU: 

