import keras_cv
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.layers import Rescaling

batch_size = 8
epochs = 100
lr = 0.0005

#load data
train_ds = image_dataset_from_directory(
  "/mnt/s/Pfad/Zu/Trainingsdaten",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    interpolation="bilinear",
    shuffle=True,
    image_size=(256,256))

test_ds = image_dataset_from_directory(
  "/mnt/s/Pfad/Zu/Testdaten",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    interpolation="bilinear",
    shuffle=True,
    image_size=(256,256))

print(train_ds)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create a model using a pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_s_imagenet",
    input_shape=(256,256,3)
)
model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=2,
    activation="softmax"
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)

model.summary()

curr_time = datetime.today().strftime('%d_%m_%Y_%H_%M_%S')

# Speicherort festlegen!
model_save_path = "/mnt/s/Pfad/Zu/Speicherort/Checkpoint_EfficientNetV2_S_Model_"+curr_time+"_epochs_"+str(epochs)+"_lr_"+str(lr)+"_bs_"+str(batch_size)

os.mkdir(model_save_path)

checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(
        train_ds,
        workers = 12,
        epochs = epochs,
        validation_data=test_ds,
        shuffle=True,
        callbacks=[checkpoint])

#Plot the Training history
fig, ax1 = plt.subplots()
ax1.plot(history.history["loss"], label="Trainingsfehler", color="orange")
ax1.plot(history.history["val_loss"], label="Testfehler", color="tomato")
ax1.set_xlabel("Epochen")
ax1.set_ylabel("MAE-Loss")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(history.history["accuracy"], label="Trainingsgenauigkeit", color="steelblue")
ax2.plot(history.history["val_accuracy"], label="Testgenauigkeit", color="turquoise")
ax2.legend(loc="upper right")
ax2.set_ylim(bottom=0, top=1.1)

plt.title("Trainingsverlauf EfficientNetV2-S")

ax2.set_ylabel("Genauigkeit")
fig.tight_layout()
plt.show()