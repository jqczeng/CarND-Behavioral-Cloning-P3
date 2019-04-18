# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_net]: ./writeup_pictures/nvidia_net.png "NVIDIA Net"
[original]: ./writeup_pictures/original-image.jpg "Original Image"
[Cropped]: ./writeup_pictures/cropped-image.jpg "Cropped Image"
[Center]: ./writeup_pictures/center_driving.png "Center Driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_YUV.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network modeled after the NVIDIA network.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving counter-clockwise on the track.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
My first step was to use a convolution neural network model similar to the the LeNet architecture I thought this model might be appropriate because LeNet had performed well for image classfication. Later on, the NVIDIA architecture was implemented because it is designed for self-driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I implemented a dropout.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order, to improve the driving behavior in these cases, I implemented several features as explained in the following:

###### Image Augmentation
During autonomous, the car may have a left turning bias because most of the training data involves the car turning left. A technique for minimizing left turn bias is flipping the images and taking the opposite sign of the steering measurement. The code can be found in the following block:

```python
images.append(cv2.flip(image,1))
measurements.append(measurement*-1.0)
```

###### Multiple Cameras
The simulator captures images from three cameras mounted on the car. There exists a left, right and center camera. By using all 3 cameras, there is a 3x amount of data available for traning. As well, the 3 images will teach the network how to recover from being off-center or steering away from center.

##### Converted images from BGR to YUV as outlined in the NVIDIA paper

All of these features adds additional training data for model which adds robustness to the the self-driving model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes as depicted in the visualization of the architecture from the NVIDIA paper:

![alt text][nvidia_net]

The following table summarizes the details of each layer.

| Layer (type)              |   Output Shape      |    Param # |
|:---------------------:|:------------------------|-----------:|
| Lambda          |  (160, 320, 3)      |  0         |
| Cropping2D      | (65, 320, 3)   |      0         |
| Convolution               | (31, 158, 24)  |    1824        |
| Convolution               | (14, 77, 36)   |    21636       |
| Convolution               | (5, 37, 48)    |    43248       |
| Convolution               | (3, 35, 64)    |    27712       |
| Convolution               | (1, 33, 64) |  36928      |
|Flatten    |      (2112)        |      0         |
|Dense         |     (100)          |     211300   |
|Dropout     |    (100)           |    0         |
|Dense         |     (50)           |     5050     |
|Dense        |     (10)           |     510      |
|Dense         |     (1)            |     11       |

#### 3. Creation of the Training Set & Training Process
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][Center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it were to drift to the side. I repeated this step several times around the corners to ensure that the vehicle does not steer off.

To allow the network to train faster, each image is cropped at the top and the bottom to crop out the data that isn't useful for tranining the network. The cropping effect can be seen in the pictures below.

![alt text][original]
Original Image

![alt text][cropped]
Cropped Image

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
