#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 06:25:17 2018

@author: mrpotatohead
"""
#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics


#Load the IMDB Dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    '''You can't feed lists of integers into a neural network. You have to turn 
    your lists into tensors. One way to do this is to One-hot encode your lists
    to turn them into vectors of 0s and 1s. This would mean, for instance, 
    turning the sequence [3,5] into a 10,000-dimensional vector that would be 
    all 0's except for indices 3 and 5, which would be 1's. Then you could use
    as the first layer in your network a Dense layer, capable of handling
    floating-point vector data.'''
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#Defining the Model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Compiling the Model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#Configuring the Optimizer
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy', 
#                                           metrics=['accuracy'])

#Using custom losses and metrics
#model.compile(optimzer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,
#              metrics=[metrics.binary_accuracy])


#Setting Aside a Validation Set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Training the Model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,validation_data=(x_val,y_val))

#Plotting the training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Plotting the training and validatin accuracy
plt.clf()   #Clears the figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, history_dict['acc'], 'bo', label='Training acc')
plt.plot(epochs, history_dict['val_acc'], 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()