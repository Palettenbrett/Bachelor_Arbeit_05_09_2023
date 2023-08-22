import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.utils import image_dataset_from_directory
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, Dropout, Rescaling
from keras.optimizers import Adam
from keras import layers
from keras.callbacks import ModelCheckpoint
from Patch_AE_Network import get_cnn_patch_autoencoder
from datetime import datetime

batch_size = 2048
epochs = 100
lr = 0.0005

#load train data
train_ds = image_dataset_from_directory(
    "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Patches_v4/IO",
    label_mode=None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size=(32,32))

#load test data
test_ds = image_dataset_from_directory(
    "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Patches_v4/IO_Test",
    label_mode=None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size=(32,32))

norm_layer = Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x: (norm_layer(x)))
normalized_test_ds = test_ds.map(lambda x: (norm_layer(x)))

normalized_train_ds_combined = normalized_train_ds.map(lambda x: (x, x))
normalized_test_ds_combined = normalized_test_ds.map(lambda x: (x, x))

# To subscript an tf.Dataset use iter and next function 
# tf.Dataset is a Tuple so index accordingly
# Test the normalisation
image_batch_1, image_batch_2 = next(iter(normalized_train_ds_combined))
print(image_batch_1.shape)
print(image_batch_2.shape)
print(np.min(image_batch_1[0]), np.max(image_batch_1[0]))
print(np.min(image_batch_2[0]), np.max(image_batch_2[0]))

AUTOTUNE = tf.data.AUTOTUNE

normalized_train_ds_combined = normalized_train_ds_combined.cache().prefetch(buffer_size=AUTOTUNE)
normalized_test_ds_combined = normalized_test_ds_combined.cache().prefetch(buffer_size=AUTOTUNE)

curr_time = datetime.today().strftime('%d_%m_%Y_%H_%M_%S')
model_save_path = "S:/STUDIUM/Bachelor_Arbeit/Models/Checkpoint_Patch_Model_"+curr_time+"_epochs_"+str(epochs)+"_lr_"+str(lr)+"_bs_"+str(batch_size)
os.mkdir(model_save_path)

checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, mode='min')

AutoEncoder = get_cnn_patch_autoencoder()

opt = Adam(learning_rate=lr)

AutoEncoder.compile(optimizer=opt, loss="mae")

AutoEncoder.summary()

history = AutoEncoder.fit(
        normalized_train_ds_combined,
        workers=12,
        epochs = epochs,
        validation_data=normalized_test_ds_combined,
        shuffle=True,
        callbacks=[checkpoint])

#Plot the Training history
plt.plot(history.history["loss"], label="Trainingsfehler", color="orange")
plt.plot(history.history["val_loss"], label="Testfehler", color="tomato")
plt.title("Trainingsverlauf des Patch CNN Autoencoders")
plt.xlabel("Epochen")
plt.ylabel("MAE-Loss")
plt.legend()
plt.show()