import keras_cv
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from datetime import datetime
from keras.models import load_model

batch_size = 1
pred_length = "Anzahl an Bildern"

#load data
val_ds = image_dataset_from_directory(
  "/mnt/s/Pfad/Zu/Daten",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=batch_size,
    #validation_split=0.2,
    #subset="training",
    interpolation="bilinear",
    #shuffle=True,
    seed=123,
    image_size=(256,256))

labels = np.zeros(shape=(pred_length,), dtype=np.int32)
preds = np.zeros(shape=(pred_length,), dtype=np.int32)

list_val_ds = list(val_ds.as_numpy_iterator())

model = load_model("/mnt/s/Pfad/Zu/Modell")

for i in range(pred_length):
    img, label = list_val_ds[i]
    labels[i] = np.argmax(label)
    pred = model.predict(img)
    preds[i] = np.argmax(pred)
    print(i)

con_mat = tf.math.confusion_matrix(labels=labels, predictions=preds)

con_mat = con_mat.numpy()

Praezision = (con_mat[0,0]/(con_mat[0,0]+con_mat[0,1]))*100
Trefferquote = (con_mat[0,0]/(con_mat[0,0]+con_mat[1,0]))*100
F1_Metrik = 2*((Praezision*Trefferquote)/(Praezision+Trefferquote))

print("Präzision: ",Praezision, "%")
print("Trefferquote: ", Trefferquote, "%")
print("F1-Metrik: ", F1_Metrik, "%")

class_labels = ["IO","NIO"]

fig, ax = plt.subplots()
im = ax.imshow(con_mat, cmap='Blues')

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        text = ax.text(j, i, con_mat[i, j],
                       ha="center", va="center", color="black")

ax.set_title("Konfusions Matrix EfficientNetV2_S")
ax.set_ylabel("Vorhergesagte Klasse")
ax.set_xlabel("Wahre Klasse")

plt.tight_layout()
plt.show()