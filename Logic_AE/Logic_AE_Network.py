'''
This is the new Version of the Logic AE Network made on 22.06.2023
It is trained on a new Dataset wich contains greyparts
Images are randomly zoomed, rotated and flipped to ensure a robust network
'''

from keras.models import Sequential
from keras.layers import Input, Conv2D, UpSampling2D, RandomZoom, RandomRotation, RandomFlip

def get_cnn_logic_autoencoder():
    AutoEncoder = Sequential()
    AutoEncoder.add(Input(shape=(512, 512, 3)))
    AutoEncoder.add(RandomZoom(height_factor=(0.415, 0.415), width_factor=(0.415, 0.415), fill_mode="constant")) # Zoom out of the pictures
    AutoEncoder.add(RandomFlip(mode="horizontal_and_vertical")) # Flip the pictures Horizontal and/or Vertical
    AutoEncoder.add(RandomRotation(factor=(-0.25, 0.25), fill_mode="constant")) # Rotate the pictures in a range of -2pi and +2pi
    AutoEncoder.add(Conv2D(filters=4, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=8, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=16, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=128, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=256, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=512, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=1024, kernel_size=4, padding="same", activation="relu", strides=2))
    #decoder
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=512, kernel_size=4, padding="same", activation="relu", strides=1))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=256, kernel_size=4, padding="same", activation="relu", strides=1))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=128, kernel_size=4, padding="same", activation="relu", strides=1))
    #AutoEncoder.add(Dropout(0.2))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu", strides=1))
    #AutoEncoder.add(Dropout(0.2))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu", strides=1))
    #AutoEncoder.add(Dropout(0.2))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=16, kernel_size=4, padding="same", activation="relu", strides=1))
    #AutoEncoder.add(Dropout(0.2))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=8, kernel_size=4, padding="same", activation="relu", strides=1))
    #AutoEncoder.add(Dropout(0.2))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=4, kernel_size=4, padding="same", activation="relu", strides=1))
    #AutoEncoder.add(Dropout(0.2))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=3, kernel_size=4, padding="same", activation="sigmoid", strides=1))
    return AutoEncoder