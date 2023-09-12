# motion_tracking_CV_bug_detector_2
Still Frame:
![Alt Text](./gif_frames/0007.jpg)
Motion tracker with moth classifier, red shows all motion detected, green shows motion with moth classifier (click for gif):
![Alt Text](./moth_motion_tracker_and_image_classifier.gif)
Bee Classifier, other motion detection removed (click for gif):
![Alt Text](./bee_classifier.gif)
Moth classifier, other motion detection removed (click for gif):
![Alt Text](./moth_classifier.gif)

# Project Overview
This repository contains code for running and training a motion detector for insects in a natural environment. Below is a list of high-level steps outlining the development:
 - OpenCV/NumPy motion tracker
 - Binary ResNet18 image classifier to remove false positives
 - Chip generator for labeling classifier data
 - Retraining script for fine-tuning ResNet18

# Motivation
The impetus for creating this repository/codebase was spurred from my interest in computer vision and its implementation in robotic systems. One example of applied computer vision that piqued my interest was a laserweeding application. In this application, an autonomous system uses computer vision to detect weeds and marks them for removal; thus eliminating the need for harmful herbicides. My interest is to expand on this solution to detect moving targets, such as bugs, and thereby reducing the need for pesticides in farming.

# Project Details
 - main.py = implements a motion tracker and image classifier on a user specified video. Also used for generating chips for data labeling.
 - main_with_video_save.py = a copy of main, with the ability to save frames as a jpg files
 - image_to_gif.py = converts jpg files to a gif
 - finetune_resnet = alters the ResNet18 architecture into a multiclass or binary classifier (binary in this case) using PyTorch to train a model on data gathered by the user into an image classifier
 - utils.py = a collection of functions called multiple times throughout the other files



# Additional Information
 - Back-propagation = calculates the gradient of the loss function with respect to the model's parameters, enabling the model to learn and improve its performance through iterative optimization. Calculates and uses the slopes (gradients) of derivatives with respect to the model's parameters to update those parameters during training, allowing the neural network to learn and improve its predictions. The gradients must be zeroed in the code to ensure the parameters are updated correctly to improve predictions.
 - Convolutional Neural Network = a deep learning model specialized in processing grid-like data, such as images, by using convolutional layers to automatically learn and extract hierarchical features from the input data.
 - Loss = quantifies the error or the difference between the predicted values and the actual target values in a machine learning or deep learning model.
 - Linear Regression = models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.
 - Weights = the numbers in the matrices in the model that determine what classified an image
 - The cv2.createBackgroundSubtractorMOG2() function uses the Mixture of Gaussians (MOG2) algorithm to model and update the background of a video sequence. It maintains a statistical model of each pixel's color over time. When a pixel's color significantly deviates from its modeled background, it is considered part of the foreground, allowing the algorithm to detect moving objects in a video by identifying pixels that change over time.

# Project Timeline
 - Motion tracker was the first implementation, but produced many false positives due to wind and camera movements. Attempted to remove plants by rejecting green objects from the dectector, but this was ineffective as the bugs and plants were too close in color.
 - Tried a multiclassifier for image classification, though landed on using a binary classifier of plant and not-plant for detecting bugs.
 - Made an error and forgot to add new weights after finetuning resnet so I thought it did not work, which led me to collect different data. Later on, this proved to be a benefit as I could test the project on multiple datasets.
 - Validation, deployment, and training data for image classification
