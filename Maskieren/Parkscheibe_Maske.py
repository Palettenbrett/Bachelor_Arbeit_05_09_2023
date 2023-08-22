import os
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

count = 1

dir = os.listdir("Path/Zum/Lesen/Der/Bilder")

random.shuffle(dir)

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = script_dir.replace("\\", "/")

mask = cv2.imread(script_dir + "/Parkscheibe_Maske.jpg", 1)
print(mask)
plt.imshow(mask)
plt.show()

while len(dir)>count:
    for element in dir:
        img_path = os.path.join("Path/Zum/Lesen/Der/Bilder/",element)
        img = cv2.imread(img_path, 1) / 255.0
        res = cv2.resize(img, (512, 512),
            interpolation = cv2.INTER_LINEAR)
        img_masked = mask*res
        write_path = "Path/Zum/Schreiben/Der/Bilder/" + str(count) +".bmp"
        cv2.imwrite(filename=write_path,img=img_masked)
        count += 1
