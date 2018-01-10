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
## 1. Required Files

### 1.1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [`model.py`](./model.py) containing the script to create and train the model
* [`drive.py`](./drive.py) for driving the car in autonomous mode
* [`model.h5`](./model.h5) containing a trained convolution neural network 
* [`writeup.md`](./writeup.md) the project report summarizing the results
* [`video.mp4`](./video.mp4) a video clip of the vehicle driving autonomously around the track

## 2. Quality of Code

### 2.1 Submission includes functional code

Using the Udacity provided simulator and the 
[`drive.py`](./drive.py) vehicle controller, 
the car can be driven autonomously around the track by executing:

```sh
python drive.py model.h5
```

### 2.2. Submission code is usable and readable

The [`model.py`](./model.py) file contains the code for training and saving the
convolution neural network (CNN) in the `*.h5` format.
The [`cnn_model.py`](./cnn_model.py) contains a CNN class, 
which creates Sequential Keras model -- 
the convolutional neural network with the architecture described [here](#arch).
In order to perform training and validation and with the large
number of images, we use `fit_generator` function and
the following generators: `training_generator` and `validation_generator`
that are implemented in the [`generators.py`](./generators.py) file.

The `model.py` is a command line utility, which uses `argparse` module
to pass the essential training parameters to the script.
In order to obtain the final `model.h5` the training script was
launched as follows:

```sh
./model.py -out_m model.h5 -f my_training_data/ -l_r 1.3e-4
```

When launching, one specifies 
(i) the output model file (`-out_m` flag), 
(ii) the path to the training images (`-f` flag), and 
(iii) the learning rate (`-l_r` flag). 
Other parameters may include 
(iv) batch size (`-b` flag), and 
(v) input model (`in_m` flag. In this case the input model
will be first read and then updated during training, 
allowing the so-called incremental learning).


## <a name='arch'>3. Model Architecture and Training Strategy </a>

### 3.1. An appropriate model architecture has been employed

The `CNN` class (in the [`cnn_model.py`](./cnn_model.py)) implements a convolutional
neural network architecture (the method `create_model_v1_2` creates 
the sequential Keras model). 
I implemented a variant of a model proposed by NVIDIA from 
[this](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
paper. 
The network accepts images of a road as inputs and outputs the steering angle.

The network first crops out the unnecessary parts of the image, then
downsamples (reduces resolution) and normalizes the data, and then
pass an image through five convolution, one max-pooling and four
fully connected layers to obtain the steering prediction.
Network details are elaborated [here](#final_arch).

### 3.2. Attempts to reduce overfitting in the model

First, I tried to use dropout, essentially after each convolutional
layer, but the network actually performs worse with dropout layers, 
so the final architecture I do not include dropout.
(Note, also that the NVIDIA end-to-end deep learning architecture 
does not use dropout as well).
Overall, the network performs quite well with a rather small validation
loss. 

### 3.3. Model parameter tuning

The model used an Adam optimizer, 
batch size by default is `32 * 6 = 192` images,
learning rate `1e-4`.
Please note how the batch size is calculated:
as we use left, right, and center images, and 
all these images flipped, the batch
size of images is then six times the number of rows in
the data frame (a row in the data frame hold information about
center, left, and right images).

### 3.4. Appropriate training data

Collection of the training data is essential to the get proper
network performance. I used images from center, left and right
cameras. For side images, I used a shifting factor of `0.115`,
which is added to the control angle to help the vehicle recover
back to center.

## 4. Architecture and Training Documentation

### 4.1. Solution Design Approach

First, I used `pandas` to read the log files which comes with
the recorded data. I then substitute the absolute path in the 
`pandas` data frame with the relative path for the given machine
(since I use Amazon EC2 instance for model training, and my local
machine for creating the training data, this step is essential
for using the training data on the *GPU* instance).

Second, I restore the previously trained Keras model (or create
a new one, and use `fit_generator` function to perform model
training and validation. I implemented both training and validation
generators that supply the network with batches of images, otherwise
holding the images in memory always result in `MemoryError`. Then,
after the training is finished, I store the model to the `*.h5` file
and test it back on my local machine.

### <a name='final_arch'> 4.2. Final Model Architecture </a>

The architecture of the network is shown below:

| Layer (type)                   |  Output Shape        |  Param #   |  Connected to                 |    
|:------------------------------:|:--------------------:|:----------:|:-----------------------------:|
| Network input: road images     |       (160, 320, 3)  |  0         |                               |    
| cropping2d_1 (Cropping2D)      |  (None, 65, 320, 3)  |  0         |  cropping2d_input_1[0][0]     |    
| maxpooling2d_1 (MaxPooling2D)  |  (None, 16, 80, 3)   |  0         |  cropping2d_1[0][0]           |    
| lambda_1 (Lambda)              |  (None, 16, 80, 3)   |  0         |  maxpooling2d_1[0][0]         |    
| Conv2D_l1 (Convolution2D)      |  (None, 14, 78, 24)  |  672       |  lambda_1[0][0]               |    
| Conv2D_l2 (Convolution2D)      |  (None, 12, 76, 36)  |  7812      |  Conv2D_l1[0][0]              |    
| Conv2D_l3 (Convolution2D)      |  (None, 10, 74, 48)  |  15600     |  Conv2D_l2[0][0]              |    
| Conv2D_l4 (Convolution2D)      |  (None, 8, 72, 64)   |  27712     |  Conv2D_l3[0][0]              |    
| Conv2D_l5 (Convolution2D)      |  (None, 6, 70, 64)   |  36928     |  Conv2D_l4[0][0]              |    
| maxpooling2d_2 (MaxPooling2D)  |  (None, 3, 35, 64)   |  0         |  Conv2D_l5[0][0]              |    
| flatten_1 (Flatten)            |  (None, 6720)        |  0         |  maxpooling2d_2[0][0]         |    
| dense_1 (Dense)                |  (None, 100)         |  672100    |  flatten_1[0][0]              |    
| dense_2 (Dense)                |  (None, 50)          |  5050      |  dense_1[0][0]                |    
| dense_3 (Dense)                |  (None, 10)          |  510       |  dense_2[0][0]                |    
| dense_4 (Dense)                |  (None, 1)           |  11        |  dense_3[0][0]                |    
|                                |                      |            |                               |   

First, the network accepts directly images of the road.
In the First layer (`Cropping2D`) the unnecessary part of the image is
cropped out
(i.e. sky above the road, and the bottom, which includes the part of the
car).
Then, I use the `MaxPooling2D` layer to downsample the image (or in other words
reduce image dimensions).
Next, I use the `Lambda` layer to normalize the pixel values to the
domain `[0,1]`.
I then use five successive convolutional layers (Convolution2D) 
with the `3`-by-`3` kernel and `24`, `36`, `48`, `64` and `64` filters.
Next,  I reduce the number of neurons in the fully connected layers
in the `MaxPooling2D` layer and then stack four fully connected
layers. The last layer outputs the result.


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
