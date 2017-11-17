# Part 1. Building the CNN (data is already prepared)
#%%
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#%%
classifier = Sequential()

# step 1: convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

#%%
# step 2: pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#%%
# step 3. flattening
classifier.add(Flatten())

#%%
# step 4 - Full connection
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

#%%
# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%
# Fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/work/notebooks/cnn-dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory('/work/notebooks/cnn-dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#%%

classifier.fit_generator(training_set,
                        samples_per_epoch=8000,
                        nb_epoch=25,
                        validation_data=test_set,
                        nb_val_samples=2000)

#%%
# homework: fit one image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/work/notebooks/cnn-dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))