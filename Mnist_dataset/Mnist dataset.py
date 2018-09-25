import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical 
from keras.layers import Convolution2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#Loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Coverting to float 
X_train=X_train.astype("float32")
X_test=X_test.astype("float32")

#Reshaping the images (Grayscale images not RGB) and 
#Rescaling the pixel values of black and grey which 256 pixels values 
X_train=X_train.reshape(X_train.shape[0],28,28,1)/255
X_test=X_test.reshape(X_test.shape[0],28,28,1)/255

#One-Hot Encoding 
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#Creating the model 
model=Sequential()
model.add(Convolution2D(32,3,3,input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

#Fully connected layers
model.add(Dense(output_dim=512,activation="relu"))#output_dim is the number nodes in the hidden layer
model.add(Dense(output_dim=1024,activation="relu"))
model.add(Dense(output_dim=10,activation="softmax"))#softmax function because its not a binary outcome

#Compile
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) 


#Creating a image augmentation for reducing over fitting 
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, y_train, batch_size=64)
test_generator = test_gen.flow(X_test, y_test, batch_size=64)


model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5, 
                    validation_data=test_generator, validation_steps=10000//64)