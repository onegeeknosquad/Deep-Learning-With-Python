#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:47:25 2018

@author: mrpotatohead
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import reuters

#Load the Dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#Vectorize the data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    '''To vectorize the labels, there are two possibilities: you can cast the label
    list as an integer tensor, or you can use one-hot encoding. One-hot 
    encoding is a widely used format for categorical data, also called 
    categorical encoding. In this case, one-hot encoding of the labels consists
    of embedding each label as an all-zero vector with a 1 in the place of the
    label index.'''
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

'''
Same as:
    from keras.utils.np_utils import to_categorical
    one_hot_train_labels = to_categorical(train_labels)
'''

#Define Model
model = models.Sequential()
model.add(layers.Dense(900, activation='relu', input_shape=(10000,)))
#model.add(layers.Dense(920, activation='relu'))
#model.add(layers.Dense(920, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

#Compiling the Model
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#Create Validation Set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


#Train the Model
history = model.fit(partial_x_train, partial_y_train, epochs=3, batch_size=512,
                    validation_data=(x_val, y_val))

#Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Plotting the training and validation accuracy
plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()