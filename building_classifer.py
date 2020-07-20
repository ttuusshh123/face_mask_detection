# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:14:26 2020

@author: Tushar
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input


initial_learning_rate = 1e-4
epochs = 20
batch_size = 32

IMAGE_SIZE = [224,224]

vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = IMAGE_SIZE+[3])

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

x = Dense(32, activation = 'relu')(x)

final = Dense(2, activation='softmax')(x)

model = Model(inputs = vgg.input, outputs = final)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True
)


training_set = train_datagen.flow_from_directory(
    'dataset',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

r = model.fit_generator(training_set, epochs=5)


