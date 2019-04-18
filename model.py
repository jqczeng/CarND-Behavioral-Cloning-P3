import csv
import cv2
import numpy as np
import sklearn
import os
from math import ceil

lines = []
# Load Custom Captured Data
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

# Load Sample Data
# with open('./data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)
#     for line in reader:
#         lines.append(line)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
#code for generator

def generator(samples, batch_size=32):
    num_samples = len(samples)       

    while 1: # Loop forever so the generator never terminates           
        shuffle(samples)
        correction = 0.3 # this is a parameter to tune
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            for line in batch_samples:
                # loop through each camera image at each collected instance
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/opt/data/IMG/' + filename
                    image = cv2.imread(current_path)
                    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    images.append(image_yuv)
                    steering_center = float(line[3]) 

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                measurements.extend([steering_left, steering_center, steering_right])
                
                # augmenting the data
                latest_images = images[-3:]
                latest_measures = measurements[-3:]
                for image, measurement in zip(latest_images, latest_measures):
                    images.append(cv2.flip(image,1))
                    measurements.append(measurement*-1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
                
            yield sklearn.utils.shuffle(X_train, y_train)
                              
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)                                                  

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Dropout
from keras.layers import Lambda
from keras.layers import Convolution2D, Conv2D
from keras.layers import MaxPooling2D

# LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

    # LeNet
#     model.add(Convolution2D(6,5,5,activation="relu"))
#     model.add(MaxPooling2D())
#     model.add(Convolution2D(6,5,5,activation="relu"))
#     model.add(MaxPooling2D())
#     model.add(Flatten())
#     model.add(Dense(120))
#     model.add(Dense(84))
#     model.add(Dense(1))

# Nvidia Network
model.add(Conv2D(24,(5,5),activation="relu",strides=(2,2)))
model.add(Conv2D(36,(5,5),activation="relu",strides=(2,2)))
model.add(Conv2D(48,(5,5),activation="relu",strides=(2,2)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
                steps_per_epoch=ceil(len(train_samples)/batch_size), \
                validation_data=validation_generator, \
                validation_steps=ceil(len(validation_samples)/batch_size), \
                epochs=1, verbose=1)

model.save('model_TEST.h5')
model.summary()
exit()