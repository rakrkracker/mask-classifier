# mask-classifier
A deep learning computer vision algorithm that detects if a person is wearing a face mask in real time via the webcam.

## Overview
An implementation of an efficient deep CNN, using transfer learning from the MobileNetV2 model, that classifies whether a person is wearing a face mask via webcam in realtime. The model was built using the TensorFlow/Keras libraries and reached an accuracy of about 99.6%.

## Libraries used
* TensorFlow
* Keras
* openCV (cv2)
* Numpy
* matplotlib
* math

## Dataset
For this project, 1,369 images of faces were used - 683 with mask and 686 without mask. The masked faces were created by [Prajna Bhandary](https://github.com/prajnasb/observations/tree/master/experiements/data). She used image augmentation to "stitch" mask graphics to images of faces.<br/>
![sample images](https://github.com/rakrkracker/mask-classifier/blob/master/images/faces_val.png)<br/>

## Data preperation
To generalize better, the images were augmented with rotation, zoom, axis shifts, shear and horizontal flips.
![augmented images](https://github.com/rakrkracker/mask-classifier/blob/master/images/faces_train.png)<br/>

## The model
The model is based on the MobileNetV2 architecture via transfer learning, making it efficient and able to run in realtime on a large range of devices. It's convolutional base was loaded with the 'imagenet' weights. A dense classification head was added, in addition to a global average pool to flatten the image and a dropout layer for more stable learning.<br/>
![transfer model](https://github.com/rakrkracker/mask-classifier/blob/master/images/transfer_model.png)<br/>

## Training
For the first training pass, the convolutional base was frozen. The model was trained in batches of size 32 and for 15 epochs.<br/>
![training curves](https://github.com/rakrkracker/mask-classifier/blob/master/images/learning_curve1.png)<br/>
Then, about 20% of the last layers in the convolutional model were unfrozen (34 out of 154) and another training run was initiated to fine tune the model. The model reached on overall accuracy of 99.85% - 99.3% validation and 99.7% training (with augmented images). This indicated a good fit, without under- or overfitting <br/>
![fine tuning curves](https://github.com/rakrkracker/mask-classifier/blob/master/images/learning_curve2.png)<br/>

# Demo
![Mask demo](https://github.com/rakrkracker/mask-classifier/blob/master/videos/cam_video_gif.gif)
