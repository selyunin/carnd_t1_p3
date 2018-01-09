# **Behavioral Cloning Project** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/img_1.png "Model Visualization"
[image2]: ./output_images/img_2.png "Grayscaling"
[image3]: ./output_images/img_3.png "Recovery Image"
[image4]: ./output_images/img_4.png "Recovery Image"
[image5]: ./output_images/img_5.png "Recovery Image"
[image6]: ./output_images/img_6.png "Normal Image"
[image7]: ./output_images/img_7.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [`model.py`](./model.py) containing the script to create and train the model
* [`drive.py`](./drive.py) for driving the car in autonomous mode
* [`model.h5`](./model.h5) containing a trained convolution neural network 
* [`writeup.md`](./writeup.md) the project report summarizing the results
* [`video.mp4`](./video.mp4) a video clip of the vehicle driving autonomously around the track

#### 2. Submission includes functional code

Using the Udacity provided simulator and the `drive.py` vehicle controller, 
the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the
convolution neural network (CNN) as an `*.h5` file.
The `cnn_model.py` contains a CNN class, which creates Sequential
Keras model -- the convolutional neural network with the architecture
described [here](arch).

The `model.py` is a command line utility, which uses `argparse` module
to pass the essential training parameters to the script.
In order to obtain the final `model.h5` the training script was
launched as follows:

```sh
./model.py -out_m model.h5 -f my_training_data/ -l_r 1.3e-4
```

When launching, one specifies (i) the output model file, (ii) the path to the
training images, and (iii) the learning rate. 
Other parameters may include (iv) batch size, and (v) input model 
(which will be first read and then updated during training, 
allowing the so-called incremental learning).


### <a id='arch'> Model Architecture and Training Strategy </a>

#### 1. An appropriate model architecture has been employed

The `CNN` class (in the `cnn_model.py`) implements a convolutional
neural network architecture (the method `create_model_v1_2` creates 
the sequential Keras model). 
I implemented a variant of a model proposed by NVIDIA from 
[this](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
paper. 
The network accepts images of a road and outputs the steering angle.

The architecture of the network is shown below:

| Layer (type)                   |  Output Shape        |  Param #   |  Connected to                 |    
|:------------------------------:|:--------------------:|:----------:|:-----------------------------:|
| cropping2d_1 (Cropping2D)      |  (None, 65, 320, 3)  |  0         |  cropping2d_input_1[0][0]     |    
|                                |                      |            |                               |    
| maxpooling2d_1 (MaxPooling2D)  |  (None, 16, 80, 3)   |  0         |  cropping2d_1[0][0]           |    
|                                |                      |            |                               |    
| lambda_1 (Lambda)              |  (None, 16, 80, 3)   |  0         |  maxpooling2d_1[0][0]         |    
|                                |                      |            |                               |    
| Conv2D_l1 (Convolution2D)      |  (None, 14, 78, 24)  |  672       |  lambda_1[0][0]               |    
|                                |                      |            |                               |    
| Conv2D_l2 (Convolution2D)      |  (None, 12, 76, 36)  |  7812      |  Conv2D_l1[0][0]              |    
|                                |                      |            |                               |    
| Conv2D_l3 (Convolution2D)      |  (None, 10, 74, 48)  |  15600     |  Conv2D_l2[0][0]              |    
|                                |                      |            |                               |    
| Conv2D_l4 (Convolution2D)      |  (None, 8, 72, 64)   |  27712     |  Conv2D_l3[0][0]              |    
|                                |                      |            |                               |    
| Conv2D_l5 (Convolution2D)      |  (None, 6, 70, 64)   |  36928     |  Conv2D_l4[0][0]              |    
|                                |                      |            |                               |    
| maxpooling2d_2 (MaxPooling2D)  |  (None, 3, 35, 64)   |  0         |  Conv2D_l5[0][0]              |    
|                                |                      |            |                               |    
| flatten_1 (Flatten)            |  (None, 6720)        |  0         |  maxpooling2d_2[0][0]         |    
|                                |                      |            |                               |    
| dense_1 (Dense)                |  (None, 100)         |  672100    |  flatten_1[0][0]              |    
|                                |                      |            |                               |    
| dense_2 (Dense)                |  (None, 50)          |  5050      |  dense_1[0][0]                |    
|                                |                      |            |                               |    
| dense_3 (Dense)                |  (None, 10)          |  510       |  dense_2[0][0]                |    
|                                |                      |            |                               |    
| dense_4 (Dense)                |  (None, 1)           |  11        |  dense_3[0][0]                |    
|                                                       |            |                               |   




#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
