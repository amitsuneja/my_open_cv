"""
https://github.com/iitmcvg/Content/blob/master/Sessions/Summer_School_2018/Session_1/Session_1.ipynb

"""

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def fileCheck(filepath):
    if not os.path.exists(filepath):
        print("source image file is missing")
        sys.exit()


CurrentPath = os.getcwd()
ImagePath = "\images\MessiOrignal.jpg"
SourceImgPath = CurrentPath + ImagePath
fileCheck(SourceImgPath)

img = cv2.imread(SourceImgPath, 1)
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols, cubeNum = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.title('My Image')

plt.subplot(222)
plt.imshow(dst)
plt.title('Rotated Image')

plt.show()