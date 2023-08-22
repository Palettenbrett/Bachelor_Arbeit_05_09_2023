# This script produces patch images from 512x512 images

import os
import cv2

IO_read = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Small_v4/IO/"
IO_Test_read = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Small_v4/IO_Test/"
#NIO_read = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe/Small/Dataset_v2/NIO/"
NIO_Test_read = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Small_v4/NIO_Test/"

IO_write = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Patches_v4/IO/"
IO_Test_write = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Patches_v4/IO_Test/"
#NIO_write = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe/Small/Dataset_v2_Patches/NIO/"
NIO_Test_write = "S:/STUDIUM/Bachelor_Arbeit/Datasets/Parkscheibe_DS_Patches_v4/NIO_Test/"

list_IO = os.listdir(IO_read)
list_IO_Test = os.listdir(IO_Test_read)
#list_NIO = os.listdir(NIO_read)
list_NIO_Test = os.listdir(NIO_Test_read)

# patching IO images
# for element in list_IO:
#     img_path = IO_read+element
#     img = cv2.imread(img_path, 1)
#     for row in range(16):
#         for col in range(16):
#             patch_img = img[row*32:(row+1)*32,col*32:(col+1)*32,:]
#             cv2.imwrite(filename=IO_write+str(row)+"_"+str(col)+"_"+element, img=patch_img)
print("done IO")
# patching IO_Test images
for element in list_IO_Test:
    img_path = IO_Test_read+element
    img = cv2.imread(img_path, 1)
    for row in range(16):
        for col in range(16):
            patch_img = img[row*32:(row+1)*32,col*32:(col+1)*32,:]
            cv2.imwrite(filename=IO_Test_write+str(row)+"_"+str(col)+"_"+element, img=patch_img)
print("done IO_Test")
# patching val images
# for element in list_NIO:
#     img_path = NIO_read+element
#     img = cv2.imread(img_path, 1)
#     for row in range(16):
#         for col in range(16):
#             patch_img = img[row*32:(row+1)*32,col*32:(col+1)*32,:]
#             cv2.imwrite(filename=NIO_write+str(row)+"_"+str(col)+"_"+element, img=patch_img)
# print("done NIO")
# patching val images
for element in list_NIO_Test:
    img_path = NIO_Test_read+element
    img = cv2.imread(img_path, 1)
    for row in range(16):
        for col in range(16):
            patch_img = img[row*32:(row+1)*32,col*32:(col+1)*32,:]
            cv2.imwrite(filename=NIO_Test_write+str(row)+"_"+str(col)+"_"+element, img=patch_img)
print("done NIO_Test")