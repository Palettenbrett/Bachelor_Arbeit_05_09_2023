from keras import Sequential
from keras.layers import Input, Conv2D, UpSampling2D, Dense, Reshape, UpSampling1D, RandomZoom, RandomFlip, RandomRotation
from keras.layers.reshaping import Flatten

# Patch fully CNN Autoencoder Network
def get_cnn_patch_autoencoder():
    AutoEncoder = Sequential()
    #encoder
    AutoEncoder.add(Input(shape=(32,32,3)))
    AutoEncoder.add(Conv2D(filters=4, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=5, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=6, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=6, kernel_size=4, padding="same", activation="relu", strides=2))
    AutoEncoder.add(Conv2D(filters=6, kernel_size=4, padding="same", activation="relu", strides=2))
    #decoder
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=6, kernel_size=4, padding="same", activation="relu", strides=1))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=6, kernel_size=4, padding="same", activation="relu", strides=1))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=5, kernel_size=4, padding="same", activation="relu", strides=1))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=4, kernel_size=4, padding="same", activation="relu", strides=1))
    AutoEncoder.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
    AutoEncoder.add(Conv2D(filters=3, kernel_size=4, padding="same", activation="sigmoid", strides=1))
    return AutoEncoder