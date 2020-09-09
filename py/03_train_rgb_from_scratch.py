#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the PyCon.DE 2018 talk by Jens Leitloff and Felix M. Riese.

PyCon 2018 talk: Satellite data is for everyone: insights into modern remote
sensing research with open data and Python.

License: MIT

"""
import os

from tensorflow.keras.applications.densenet import DenseNet201 as DenseNet
from tensorflow.keras.applications.vgg16 import VGG16 as VGG
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from image_functions import preprocessing_image_rgb

# variables
path_to_split_datasets = "~/Documents/Data/PyCon/RGB"
use_vgg = False
batch_size = 64

# contruct path
path_to_home = os.path.expanduser("~")
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")

# get number of classes
sub_dirs = [sub_dir for sub_dir in os.listdir(path_to_train)
            if os.path.isdir(os.path.join(path_to_train, sub_dir))]
num_classes = len(sub_dirs)

# parameters for CNN
if use_vgg:
    base_model = VGG(include_top=False,
                     weights=None,
                     input_shape=(64, 64, 3))
else:
    base_model = DenseNet(include_top=False,
                          weights=None,
                          input_shape=(64, 64, 3))
# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
# or just flatten the layers
# top_model = Flatten()(top_model)
# let's add a fully-connected layer
if use_vgg:
    # only in VGG19 a fully connected nn is added for classfication
    # DenseNet tends to overfitting if using additionally dense layers
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
# and a logistic layer
predictions = Dense(num_classes, activation='softmax')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# print network structure
model.summary()

# defining ImageDataGenerators
# ... initialization for training
train_datagen = ImageDataGenerator(
    fill_mode="reflect",
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocessing_image_rgb)

# ... initialization for validation
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_image_rgb)

# ... definition for training
train_generator = train_datagen.flow_from_directory(path_to_train,
                                                    target_size=(64, 64),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
print(train_generator.class_indices)

# ... definition for validation
validation_generator = test_datagen.flow_from_directory(
    path_to_validation,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical')

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_rgb_from_scratch." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=50,
                             mode='max')
model.fit(
    train_generator,
    steps_per_epoch=1000,
    epochs=10000,
    callbacks=[checkpointer, earlystopper],
    validation_data=validation_generator,
    validation_steps=500)
