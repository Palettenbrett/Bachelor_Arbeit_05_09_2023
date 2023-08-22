# This script is for training an Autoencoder for logical defect detection

import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import image_dataset_from_directory
from keras.layers import Rescaling
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from Logic_AE_Network import get_cnn_logic_autoencoder

batch_size = 8
epochs = 100
lr = 0.0005

#load train data IO
train_ds = image_dataset_from_directory(
  "Path/Zu/Trainingsbildern",
    label_mode=None,
    color_mode = "rgb",
    batch_size=batch_size,
    image_size=(512,512))

#load test data IO
test_ds = image_dataset_from_directory(
    "Path/Zu/Testbildern",
    label_mode=None,
    color_mode = "rgb",
    batch_size=batch_size,
    image_size=(512,512))

norm_layer = Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x: (norm_layer(x)))
normalized_test_ds = test_ds.map(lambda x: (norm_layer(x)))

normalized_train_ds_combined = normalized_train_ds.map(lambda x: (x, x))
normalized_test_ds_combined = normalized_test_ds.map(lambda x: (x, x))

image_batch_1, image_batch_2 = next(iter(normalized_test_ds_combined))
print(image_batch_1.shape)
print(image_batch_2.shape)
print(np.min(image_batch_1[0]), np.max(image_batch_1[0]))
print(np.min(image_batch_2[0]), np.max(image_batch_2[0]))

AUTOTUNE = tf.data.AUTOTUNE

normalized_train_ds_combined = normalized_train_ds_combined.cache().prefetch(buffer_size=AUTOTUNE)
normalized_test_ds_combined = normalized_test_ds_combined.cache().prefetch(buffer_size=AUTOTUNE)

curr_time = datetime.today().strftime('%d_%m_%Y_%H_%M_%S')
model_save_path = "S:/STUDIUM/Bachelor_Arbeit/Models/Checkpoint_Logic_Model_"+curr_time+"_epochs_"+str(epochs)+"_lr_"+str(lr)+"_bs_"+str(batch_size)
os.mkdir(model_save_path)

checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

AutoEncoder = get_cnn_logic_autoencoder()

AutoEncoder.summary()

opt = Adam(learning_rate=lr)

AutoEncoder.compile(optimizer=opt, loss="mae")

history = AutoEncoder.fit(
        normalized_train_ds_combined,
        workers = 12,
        epochs = epochs,
        validation_data=normalized_test_ds_combined,
        shuffle=True,
        callbacks=[checkpoint])

#Plot the Training history
fig, ax1 = plt.subplots()
ax1.plot(history.history["loss"], label="Trainingsfehler", color="orange")
ax1.plot(history.history["val_loss"], label="Testfehler", color="tomato")
ax1.set_xlabel("Epochen")
ax1.set_ylabel("MAE-Loss")

plt.title("Trainingsverlauf des Logik CNN Autoencoders")
plt.legend()
plt.show()

#Predict
pred = AutoEncoder.predict(normalized_test_ds, batch_size=1, steps=10)
print(pred.shape)

pred = tf.reduce_mean(pred, axis=3, keepdims=True)
print(pred.shape)

for i in range(10):
  plt.imshow(pred[i,:,:,:], cmap="gray")
  plt.show()