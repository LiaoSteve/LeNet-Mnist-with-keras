# http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
# https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48
import os 
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
print(os.listdir('/'))
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)  
X_test = X_test.reshape(-1, 28, 28, 1)      
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=5e-4))
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                                patience=3, 
                                verbose=1, 
                                factor=0.2, 
                                min_lr=1e-6)

os.makedirs('./log',exist_ok=True)
log = keras.callbacks.TensorBoard(log_dir='./log',
                                         histogram_freq=1,
                                         embeddings_freq=0,
                                         embeddings_layer_names=None)

datagen = ImageDataGenerator(
            rotation_range=10, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            shear_range=0.1,
            zoom_range=0.2)
datagen.fit(X_train)                                         

model.fit_generator(datagen.flow(X_train, y_train, batch_size=100), steps_per_epoch=len(X_train)/100, 
                    epochs=2, validation_data=(X_test, y_test), callbacks=[reduce_lr,log])

score = model.evaluate(X_test, y_test, batch_size=32)
print('score: ',score)
model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it