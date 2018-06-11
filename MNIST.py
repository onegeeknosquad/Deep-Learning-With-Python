# -*- coding: utf-8 -*-
"""
Russell J. Adams
6-10-2018

Neural Network that learns to classify handwritten digits using the MNIST
dataset.
"""

from keras.utils import to_categorical
from keras.datasets import mnist
from keras import models
from keras import layers

#Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Network Architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

#Compilation Step
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Preparing Image Data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#Prepare the Labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Train the Network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#Run the Model Versus the Test Set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)