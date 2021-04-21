import numpy as np
import pandas as pd
import tensorflow as tf

#Import data 
labels = np.load('./data/labels_num.npy')
data = np.load('./data/text_embed.npy')

#Shuffle data
order = np.random.permutation(np.arange(len(data)))
data, labels = data[order], labels[order]
train_x, test_x = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
train_y, test_y = labels[:int(len(data)*0.8)], labels[int(len(data)*0.8):]




