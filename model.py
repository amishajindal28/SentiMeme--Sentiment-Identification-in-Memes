import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense, LSTM
from keras.optimizers import Adam, SGD

#Import data 
labels = np.load('./data/labels_num.npy')
data = np.load('./data/text_embed.npy')

#Shuffle and split data
order = np.random.permutation(np.arange(len(data)))
data, labels = data[order], labels[order]
data = data.reshape((len(data), 768, 1))
labels = labels.reshape((len(labels),1))
train_x, test_x = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
train_y, test_y = labels[:int(len(data)*0.8)], labels[int(len(data)*0.8):]
train_x, valid_x = train_x[:int(len(train_x)*0.8)], train_x[int(len(train_x)*0.8):]
train_y, valid_y = train_y[:int(len(train_y)*0.8)], train_y[int(len(train_y)*0.8):]

#Model definition
def model_(opt):
    model = tf.keras.Sequential()

    # Must define the input shape in the first layer of the neural network
    model.add(Conv1D(64, (8), activation='relu',input_shape=(768, 1)))
    model.add(Conv1D(64, (8), activation='relu'))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))

    model.add(Conv1D(64, (16), activation='relu'))
    model.add(Conv1D(64, (16), activation='relu'))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))

    model.add(Conv1D(64, (32), activation='relu'))
    model.add(Conv1D(64, (32), activation='relu'))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))

    model.add(Conv1D(64, (64), activation='relu'))
    model.add(Conv1D(64, (64), activation='relu'))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))

    model.add(Conv1D(64, (128), activation='relu'))
    model.add(Conv1D(64, (128), activation='relu'))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))

    model.add(LSTM(3))
    model.add(Dense(units=128,activation="relu"))
    model.add(Dense(units=1, activation="tanh"))
    
    model.compile(loss='mean_squared_error',
             optimizer=opt,
             metrics=['mean_squared_error'])

    return model

#Initialize optimization
opt = Adam(learning_rate=0.00075)

#Model initialization
model = model_(opt)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

#Train the network from scratch
history = model.fit(  train_x, train_y, batch_size=64, epochs=25,
            validation_data=(valid_x, valid_y),
            callbacks = [callback])

#TODO
'''
1. Calculation of F1 Score
2. Check if the current arquitechture is working well
3. Replicate the arquitechture of the paper and see if it works
'''


