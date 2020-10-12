# General libraries

import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt

# Deep learning libraries
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


def cnn_model(img_dims):
    # Input layer
    X_input = Input(shape=(img_dims, img_dims, 3))

    # First convolutional block
    X = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(X_input)
    X = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = MaxPool2D(pool_size=(2, 2))(X)

    # Second convolutional block
    X = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(2, 2))(X)

    # Third convolutional block
    X = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(2, 2))(X)

    # Fourth convolutional block
    X = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Dropout(rate=0.2)(X)

    # Fifth convolutional block
    X = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Dropout(rate=0.2)(X)

    # Fully-connected layer
    X = Flatten()(X)
    X = Dense(units=256, activation='relu')(X)
    X = Dropout(rate=0.6)(X)
    X = Dense(units=128, activation='relu')(X)
    X = Dropout(rate=0.5)(X)
    X = Dense(units=64, activation='relu')(X)
    X = Dropout(rate=0.3)(X)

    # Output layer
    output = Dense(units=1, activation='sigmoid')(X)

    # Create model and compile
    model = Model(inputs=X_input, outputs=output)
    
    return model

