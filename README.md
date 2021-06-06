# mask-classifier
A deep learning computer vision algorithm that detects if a person is wearing a face mask in real time via the webcam.

## Overview
A video stream from the webcam is captured and put through a pretrained deep CNN to classify whether a person is wearing a face mask.
The nn is based on the MobileNetV2 architecture as a base, loaded with the 'imagenet' weights, and trained via transfer learning through a dense classification head. This makes the model lightweight and able to run in realtime on many devices.
The base was frozen during initial training, then about 20% of the top layers were unfrozen for fine tuning.
The model reached an accuracy of about 99.5%.
