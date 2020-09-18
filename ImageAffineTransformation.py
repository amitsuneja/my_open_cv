"""
https://www.tutorialspoint.com/dip/concept_of_edge_detection.htm

"""

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Affine Transformation - In Affine transformation, all parallel lines in the original image 
will still be parallel in the output image.To find the transformation matrix, we need 
three points from input image and their corresponding locations in the output image. 
Then cv2.getAffineTransform will create a 2×3 matrix which is to be passed to 
cv2.warpAffine.

Syntax: cv2.getPerspectiveTransform(src, dst)
Parameters:-
src: Coordinates of quadrangle vertices in the source image.
dst: Coordinates of the corresponding quadrangle vertices in the destination image.

Syntax: cv2.warpAffine(src, M, dsize, dst, flags, borderMode, borderValue)
Parameters:-
src: input image.
dst: output image that has the size dsize and the same type as src.
M: transformation matrix.
dsize: size of the output image.
flags: combination of interpolation methods (see resize() ) and the optional flag
WARP_INVERSE_MAP that means that M is the inverse transformation (dst->src).
borderMode: pixel extrapolation method; when borderMode=BORDER_TRANSPARENT, 
it means that the pixels in the destination image corresponding to the “outliers” in the 
source image are not modified by the function.
borderValue: value used in case of a constant border; by default, it is 0.



Translation is the shifting of object’s location. If you know the shift in (x,y) 
direction, let it be (T_x,T_y), you can create the transformation matrix M as follows:

M =[1 0 T_x0 1 T_y]

You can take make it into a Numpy array of type np.float32 and pass it into cv2.warpAffine() 
function. See below example2(output2) for a shift of (100,50):
"""


def fileCheck(filepath):
    if not os.path.exists(filepath):
        print("source image file is missing")
        sys.exit()


CurrentPath = os.getcwd()
ImagePath = "\images\MessiOrignal.jpg"
SourceImgPath = CurrentPath + ImagePath
fileCheck(SourceImgPath)

img = cv2.imread(SourceImgPath, 1)
rows, cols, cubeNum = img.shape

pts1 = np.float32([[50, 50],[200, 50],[50, 200]])
pts2 = np.float32([[10, 100],[200, 50],[100, 250]])
M1 = cv2.getAffineTransform(pts1, pts2)
M2 = np.float32([[1,0,100],[0,1,50]])
dst1 = cv2.warpAffine(img, M1, (cols, rows))
dst2 = cv2.warpAffine(img, M2, (cols, rows))
print("pts1=", pts1)
print("pts2=", pts2)
print("M1=", M1)
print("M2=", M2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.set_title("Input")
ax1.imshow(img)
ax2.set_title("Output1")
ax2.imshow(dst1)
ax3.set_title("Output2")
ax3.imshow(dst2)
plt.show()

