import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import os
import time

AutoEncoder = load_model("S:/STUDIUM/Bachelor_Arbeit/Models/Checkpoint_Logic_Model_03_08_2023_11_21_04_epochs_100_lr_0.0005_bs_8")
AutoEncoder.summary()

filepath_good = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Small_v4/IO_Test/"
filepath_bad = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Small_v4/NIO_Test/"

img_array = np.ndarray(shape=(1,512,512,3), dtype=np.float32)

score_array_good = np.ndarray(shape=(600,1), dtype=np.float32)
score_array_bad = np.ndarray(shape=(600,1), dtype=np.float32)

top_val_good = np.ndarray(shape=(600,1), dtype=np.float32)
top_val_bad = np.ndarray(shape=(600,1), dtype=np.float32)

# lim_val = 0.4
# top_vals = 0

# get list of pictures good in sorted order
file_good_sorted = os.listdir(filepath_good)
#file_good_sorted = []

# name_count = 1

# while name_count <= 150:
#     for element in file_good:
#         split = element.split(".")[0]
#         if int(split) == name_count:
#             file_good_sorted.append(element)
#             name_count += 1
#             break

# print(file_good_sorted)

# # get list of pictures bad in sorted order
file_bad_sorted = os.listdir(filepath_bad)
#file_bad_sorted = []

# name_count = 1

# while name_count <= 150:
#     for element in file_bad:
#         split = element.split(".")[0]
#         if int(split) == name_count:
#             file_bad_sorted.append(element)
#             name_count += 1
#             break

# print(file_bad_sorted)

# Predictions ------------------------------------------------

count = 0

for element in file_good_sorted:
    img_path = filepath_good+element
    img = cv2.imread(img_path, 1) / 255
    img_array[0,:,:,:] = img[:,:,:] 

    pred = AutoEncoder.predict(img_array, batch_size=1)

    anomaly_map = tf.math.abs(img_array - pred)[0,:,:,0]

    # anomaly_vec = tf.reshape(anomaly_map, shape=(512**2,1))
    # anomaly_vec = tf.sort(anomaly_vec, axis=0, direction="DESCENDING")

    # anomaly_vec = tf.get_static_value(anomaly_vec)

    # for i in range(len(anomaly_vec[:,0])):
    #     if anomaly_vec[i,0] < lim_val:
    #         top_val_good[count,0] = i
    #         break

    
    
    # plt.imshow(anomaly_map, cmap="RdYlGn_r")
    # plt.show()

    score_array_good[count,0] = tf.get_static_value(tf.reduce_mean(anomaly_map))


    count += 1
    if count == 600:
        break

count = 0

for element in file_bad_sorted:
    img_path = filepath_bad+element
    img = cv2.imread(img_path, 1) / 255
    img_array[0,:,:,:] = img[:,:,:] 

    pred = AutoEncoder.predict(img_array, batch_size=1)
    
    anomaly_map = tf.math.abs(img_array - pred)[0,:,:,0]

    # anomaly_vec = tf.reshape(anomaly_map, shape=(512**2,1))
    # anomaly_vec = tf.sort(anomaly_vec, axis=0, direction="DESCENDING")

    # anomaly_vec = tf.get_static_value(anomaly_vec)

    # for i in range(len(anomaly_vec[:,0])):
    #     if anomaly_vec[i,0] < lim_val:
    #         top_val_bad[count,0] = i
    #         break
    


    # plt.imshow(anomaly_map, cmap="RdYlGn_r")
    # plt.show()

    score_array_bad[count,0] = tf.get_static_value(tf.reduce_mean(anomaly_map))

    fig, axes = plt.subplots(1,3)
    # Display the first image on the first subplot
    axes[0].imshow(img_array[0,:,:,:])
    axes[0].axis('off')
    axes[0].set_title("Original")

    # Display the second image on the second subplot
    axes[1].imshow(pred[0,:,:,:], cmap="Reds")
    axes[1].axis('off')
    axes[1].set_title("Rekonstruktion")

    # Display the third image on the third subplot
    axes[2].imshow(anomaly_map, cmap="RdYlGn_r")
    axes[2].axis('off')
    axes[2].set_title("Anomaly Map")

    plt.suptitle("Bildausgabe des Logik CNN Autoencoders (IO)")

    anomaly_vector = tf.reshape(anomaly_map, shape=(512**2,1)).numpy()

    # axes[3].hist(anomaly_vector, bins=250)
    # axes[3].set_ylim([0,1000])

    # axes[3].set_xlim([0,1])
    # axes[3].axis('on')

    plt.subplots_adjust(wspace=0.1)

    plt.show()


    count += 1
    if count == 600:
        break



plt.plot(score_array_good, label="Fehlerwerte IO-Teile")
plt.plot(score_array_bad, label="Fehlerwerte NIO-Teile")
plt.xlabel("")
plt.ylabel("MAE-Werte")
plt.legend()
plt.title("IO/NIO Rekonstruktionsfehler des Logik CNN Autoencoder")
plt.show()

plt.hist(score_array_bad, bins=100, alpha=1, label="NIO-Teile", color="darkorange")
plt.hist(score_array_good, bins=100, alpha=1, label="IO-Teile", color="steelblue")
plt.xlabel("MAE Fehlerwerte pro Pixel")
plt.ylabel("Fehlerhäufigkeit")
plt.title("MAE-Fehlerhäufigkeit bei Rekonstruktionen")
plt.legend()
plt.show()



#simple metric for accruacy

true_pos_array = np.ndarray(shape=(1000,1), dtype=np.float32) 
false_pos_array = np.ndarray(shape=(1000,1), dtype=np.float32)
false_neg_array = np.ndarray(shape=(1000,1), dtype=np.float32) 
true_neg_array = np.ndarray(shape=(1000,1), dtype=np.float32) 

# check IO accruacy
i = 0
index = 0
while i < 1:
    true_pos = 0
    true_neg = 0
    for element in score_array_good[:,0]:
        if element < i:
            true_pos += 1
        else:
            true_neg += 1
    true_pos_array[index,0] = true_pos
    true_neg_array[index,0] = true_neg
    i += 0.001
    index += 1

# check NIO accruacy
i = 0
index = 0
while i < 1:
    false_neg = 0
    false_pos = 0
    for element in score_array_bad[:,0]:
        if element > i:
            false_neg += 1
        else:
            false_pos += 1
    false_neg_array[index,0] = false_neg
    false_pos_array[index,0] = false_pos
    i += 0.001
    index += 1

acc_array = np.add(true_pos_array,false_neg_array)
acc_array = acc_array - 150

val_max_acc = np.argmax(acc_array)
val_max_acc = val_max_acc / 1000

rp = np.count_nonzero(score_array_good < val_max_acc)
rn = np.count_nonzero(score_array_bad >= val_max_acc)
fp = np.count_nonzero(score_array_bad < val_max_acc)
fn = np.count_nonzero(score_array_good >= val_max_acc)

'''
        true
        IO    NIO  
pred    
IO      rp    fp
NIO     fn    rn

'''

Correct_Class = (rp+rn)/(rp+rn+fp+fn)
Precision = rp/(rp+fp)
Recall = rp/(rp+fn)
F1_Score = 2*((Precision*Recall)/(Precision+Recall))

print("Correct Classified: ",(np.round(Correct_Class, decimals=3))*100," %")
print("Genauigkeit: " ,(np.round(Precision, decimals=3))*100," %")
print("Trefferquote: ",(np.round(Recall, decimals=3))*100," %")
print("F1 Score: ",(np.round(F1_Score, decimals=3))*100," %")
# Confusion Matrix--------------------------

confusion_matrix = np.array([[rp, fp],
                             [fn, rn]])

class_labels = ['IO', 'NIO']

fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap='Blues')

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        text = ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center", color="black")

ax.set_title("Konfusions Matrix Logik CNN Autoencoder")
ax.set_ylabel("Vorhergesagte Klasse")
ax.set_xlabel("Wahre Klasse")

plt.tight_layout()
plt.show()