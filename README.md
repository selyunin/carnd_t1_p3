# Behaviorial Cloning Project 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

The repository contains files for the Behavioral Cloning Project.


In this project, we use what deep learning framework `keras` to build
a convolutional neural network that is capable to clone driving behavior.
We trained, validated and tested a model.
The model will output a steering angle to an autonomous vehicle.


Udacity has provided a simulator where one can steer a car around 
a track for data collection. I used image data and steering angles to
train the Keras model that is capable to autonomously steer 
the vehicle around the track.


I also created a detailed writeup of the project, describing how the 
model has been created and addressing rubric points. 
Check out the writeup [here](./writeup.md).


To meet specifications, the project contains the following five files: 
* [`model.py`](./model.py)   (script used to create and train the model)
* [`drive.py`](./drive.py)   (script to drive the car - feel free to modify this file)
* [`model.h5`](./model.h5)   (a trained Keras model)
* [`writeup.md`](./writeup.md) (the project report file)
* [`video.mp4`](./video.mp4)  (a video clip of the vehicle driving autonomously around the track for at least one full lap)


## The Project


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

