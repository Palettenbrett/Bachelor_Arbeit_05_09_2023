import os
import cv2
import random

count = 1

dir = os.listdir("S:/STUDIUM/Bachelor_Arbeit/Datasets/NanoBlades_Validation_Large/NIO")
#print(dir)
random.shuffle(dir)
#print(dir)

while len(dir)>count:
    for element in dir:
        img_path = os.path.join("S:/STUDIUM/Bachelor_Arbeit/Datasets/NanoBlades_Validation_Large/NIO/",element)
        img = cv2.imread(img_path, 1)
        res = cv2.resize(img, (512, 512),
             interpolation = cv2.INTER_LINEAR)
        write_path = "S:/STUDIUM/Bachelor_Arbeit/Datasets/NanoBlades_Validation_Small/NIO/" + str(count) +".bmp"
        cv2.imwrite(filename=write_path,img=res)
        print(count)
        count += 1
