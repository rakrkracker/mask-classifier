# mask-classifier
A deep learning computer vision algorithm that detects if a person is wearing a face mask in real time via the webcam.

## Overview
An implementation of an efficient deep CNN that classifies whether a person is wearing a face mask via webcam in realtime.

## Dataset
For this project, 1,369 images of faces were used - 683 with mask and 686 without mask. The masked faces were created by [Prajna Bhandary](https://github.com/prajnasb/observations/tree/master/experiements/data). She used image augmentation to "stitch" mask graphics to images faces.<br/>
![sample images](https://github.com/rakrkracker/mask-classifier/blob/master/images/faces_val.png)

## Data preperation
To generalize better, the images were augmented with rotation, zoom, axis shifts, shear and horizontal flips.
![augmented images](https://github.com/rakrkracker/mask-classifier/blob/master/images/faces_train.png)

## The model
The model is based on the MobileNetV2 architecture, making it efficient and able to run in realtime on a large range of devices. It's convolutional base was loaded with the 'imagenet' weights. A dense classification head was added, in addition to a global average pool to flatten the image and a dropout layer for more stable learning.

## Training
For the first training pass, the convolutional base was frozen. The model was trained in batches of size 32 and for 15 epochs. Then, about 20% of the last layers in the convolutional model were unfrozen (34 out of 154) and another training run was initiated to fine tune the model. The model reached on overall accuracy of 99.6% - 99.3% validation and 99.9% training (with augmented images). <br/>
![training curves](https://github.com/rakrkracker/mask-classifier/blob/master/images/learning_curve1.png)
![fine tuning curves](https://github.com/rakrkracker/mask-classifier/blob/master/images/learning_curve2.pdf)
