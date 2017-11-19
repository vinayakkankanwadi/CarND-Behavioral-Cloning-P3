import csv
import cv2
import numpy as np
import tensorflow as tf

lines = []
with open('./data-track1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './data-track1/IMG/'+filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	
X_train = np.array(images)
y_train = np.array(measurements)

print(len(X_train))
print(X_train.shape)
print(len(y_train))
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Convolution2D, Cropping2D, Conv2D

def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    model.add(Conv2D(24,(5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), strides=(1, 1))) 
    model.add(Activation('relu'))

    model.add(Flatten())    

    model.add(Dense(100))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse") 
    
    return model

model = get_model()
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=7)
model.save('model.h5')
