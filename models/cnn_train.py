# Move data from images to images/train or images/test: 
import shutil
from collections import defaultdict
import json
from pathlib import Path
import os

from pathlib import Path

download_dir = Path('/content/')

# split_dataset(download_dir/'food-101')

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Image augmentations
example_generator = ImageDataGenerator(
    rescale=1 / 255.,           # normalize pixel values between 0-1
    vertical_flip=True,         # vertical transposition
    horizontal_flip=True,       # horizontal transposition
    rotation_range=90,          # random rotation at 90 degrees
    height_shift_range=0.3,     # shift the height of the image 30%
    brightness_range=[0.1, 0.9] # specify the range in which to decrease/increase brightness
)

from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(
    rescale=1/255.,              # normalize pixel values between 0-1
    brightness_range=[0.1, 0.7], # specify the range in which to decrease/increase brightness
    width_shift_range=0.5,       # shift the width of the image 50%
    rotation_range=90,           # random rotation by 90 degrees
    horizontal_flip=True,        # 180 degree flip horizontally
    vertical_flip=True,          # 180 degree flip vertically
    validation_split=0.15        # 15% of the data will be used for validation at end of each epoch
)

import os

class_subset = sorted(os.listdir(str(download_dir/'food-101/images')))[:10]

BATCH_SIZE = 32

traingen = train_generator.flow_from_directory(
    str(download_dir/'food-101/train'),
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_subset,
    subset='training',
    shuffle=True,
    seed=42
)

validgen = train_generator.flow_from_directory(
    str(download_dir/'food-101/test'),
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_subset,
    subset='validation',
    shuffle=True,
    seed=42
)

import tensorflow as tf
print(tf.test.is_gpu_available())


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.regularizers import l1_l2

model = Sequential()

#### Input Layer ####
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 activation='relu', input_shape=(128, 128, 3)))

#### Convolutional Layers ####
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))  # Pooling
model.add(Dropout(0.2)) # Dropout

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (5,5), padding='same', activation='relu'))
model.add(Conv2D(512, (5,5), activation='relu'))
model.add(MaxPooling2D((4,4)))
model.add(Dropout(0.2))

#### Fully-Connected Layer ####
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(class_subset), activation='softmax'))

print(model.summary()) # a handy way to inspect the architecture

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from livelossplot import PlotLossesKeras

steps_per_epoch = traingen.samples // BATCH_SIZE
val_steps = validgen.samples // BATCH_SIZE

n_epochs = 100

optimizer = RMSprop(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Saves Keras model after each epoch
checkpointer = ModelCheckpoint(filepath='img_model.weights.best.hdf5', 
                               verbose=1, 
                               save_best_only=True)

# Early stopping to prevent overtraining and to ensure decreasing validation loss
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                        #    restore_best_weights=True,
                           mode='min')

# tensorboard_callback = TensorBoard(log_dir="./logs")

# Actual fitting of the model
history = model.fit_generator(traingen,
                    epochs=n_epochs, 
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validgen,
                    validation_steps=val_steps,
                    callbacks=[early_stop, checkpointer, PlotLossesKeras()],
                    verbose=False)

# test_generator = ImageDataGenerator(rescale=1/255.)

# testgen = test_generator.flow_from_directory(str(download_dir/'food-101/test'),
#                                              target_size=(128, 128),
#                                              batch_size=1,
#                                              class_mode=None,
#                                              classes=class_subset, 
#                                              shuffle=False,
#                                              seed=42)

# model.load_weights('img_model.weights.best.hdf5')

# predicted_classes = model.predict(testgen)

# class_indices = traingen.class_indices
# class_indices = dict((v,k) for k,v in class_indices.items())
# true_classes = testgen.classes

