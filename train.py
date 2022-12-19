# import required libraries
import cv2
import numpy as np
import os
import tensorflow as tf
import random

# the folder where the images are
path_with = "data/with"
path_without = "data/without"

# reads images and turns them into a list (img, 0 or 1).
def to_list(path, k):
    arr = []
    for i in os.listdir(path):
        img = cv2.imread(path + "/" + i)
        img = img / 255
        img = np.array([img, k])
        arr.append(img)
    return arr

# upload images without mask
data_without = to_list(path_without, 0)
# upload images with mask
data_with = to_list(path_with, 1)
# we divide the first 1500 images from each into training data
training_data=data_without[:1500]+data_with[:1500]
# we divide last 250 images from each into test data
test_data=data_with[1500:]+data_with[1500:]
#shuffling the data
random.shuffle(training_data)
random.shuffle(test_data)
# separate test and training data
x_train=[i[0] for i in training_data]
y_train=[i[1] for i in training_data]
x_test=[i[0] for i in test_data]
y_test=[i[1] for i in test_data]
# list to numpy array
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
# creating model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((100,100,3)),
    tf.keras.layers.Conv2D(10,(3,3),padding="same",activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),   # 50 50 10
    tf.keras.layers.Conv2D(5,(3,3),padding="same",activation="relu"),# 50 50 50
    tf.keras.layers.MaxPool2D((2,2)), # 25 25 50
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(625,activation="relu"),
    tf.keras.layers.Dense(125,activation="relu"),
    tf.keras.layers.Dense(25,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
loss = tf.keras.losses.BinaryCrossentropy()
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim,loss=loss,metrics=["accuracy"])
# training
model.fit(x_train,y_train,epochs=7)
# evaluate
model.evaluate(x_test,y_test)
# saving the model
model.save("mask.h5")
