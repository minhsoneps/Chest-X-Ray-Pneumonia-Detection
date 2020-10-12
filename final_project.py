# Import general libraries
import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt

# Import tensorflow and keras 
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import data_processing
import cnn_model


# Setting seeds
seed = 232
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Set input path
input_path = 'data/'

# Set the image dimension, number of epochs, and batch size
img_dims = 150
epochs = 30
batch_size = 32

# Process the data 
train_gen, test_gen, test_data, test_labels = data_processing.data_processing(img_dims, batch_size)

# Create model and compile
model = cnn_model.cnn_model(img_dims)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')


# Train the model
cnn = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples // batch_size, epochs=epochs, validation_data=test_gen, 
           validation_steps=test_gen.samples // batch_size, callbacks=[checkpoint, lr_reduce])

# Plot the model accuracy with respect to number of epochs
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()
for i, metric in enumerate(['accuracy', 'loss']):
    ax[i].plot(cnn.history[metric])
    ax[i].plot(cnn.history['val_' + metric])
    ax[i].set_title('Model {}'.format(metric))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(metric)
    ax[i].legend(['train', 'val'])
plt.savefig('Model accuracy and Model loss.jpg')
plt.show()


# Print out confusion matrix, accuracy, precision, recall, F1-score and train accuracy
preds = model.predict(test_data)
acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX:')
print(cm)
print('\nTEST METRICS:')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy:',acc,'%')
print('Precision:',precision,'%')
print('Recall:',recall,'%')
f1 = 2*precision*recall/(precision+recall)
print('F1-score:',f1)

print('\nTRAINING METRIC:')
train_acc = (cnn.history['accuracy'][-1])*100
print('Training accuracy:',train_acc,'%')