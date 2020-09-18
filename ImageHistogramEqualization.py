"""
In Statistics, Histogram is a graphical representation showing a visual impression
of the distribution of data.

In image processing, histograms are associated with the intensity values of pixels.
For an 8 bit greyscale, we have 256 different bins (0-255).





"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def dispImg(img,windows_name="openCVPic"):
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fileCheck(filepath):
    if not os.path.exists(filepath):
        print("source image file is missing")
        sys.exit()

def histogramOfGrayImage(img):
    """
    cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

    images : it is the source image of type uint8 or float32.
    it should be given in square brackets, ie, "[img]".

    channels : it is also given in square brackets.
    it is the index of channel for which we calculate histogram.
    For grayscale image, its value is [0] and color image, you can pass [0], [1] or [2]
    to calculate histogram of blue, green or red channel respectively.

    mask : mask image. To find histogram of full image, it is given as "None".
    But if you want to find histogram of particular region of image, you have
    to create a mask image for that and give it as mask. (I will show an example later.)

    histSize : this represents our BIN count. Need to be given in square brackets.
    For full scale, we pass [256].

    ranges : this is our RANGE. Normally, it is [0,256].
    """
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Note: Histograms can be calculated using numpy functions also. They will
    # also the same output as the openCV functions. But the OpenCV functions are
    # more faster (40X) than numpy functions
    # Code in Numpy
    # hist, bins = np.histogram(img.ravel(), 256, [0,256])
    plt.plot(histogram)
    plt.xlabel('Pixel intensity values (0 - 255)')
    plt.ylabel('No of pixels')
    plt.xlim([0, 256])
    plt.title('Image Histogram')
    plt.show()


def histogramOfAllChannelsColorImage(img):
    color = {'b', 'g', 'r'}
    for i, col in enumerate(color):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histogram, color=col)
        plt.xlim([0, 256])
    plt.show()




if __name__=="__main__":
    CurrentPath = os.getcwd()
    ImagePath = "\images\MessiOrignal.jpg"
    SourceImgPath = CurrentPath + ImagePath
    fileCheck(SourceImgPath)
    img_bgr = cv2.imread(SourceImgPath, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rbg =cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(img_rbg)
    plt.show()
    histogramOfGrayImage(img_gray)
    histogramOfAllChannelsColorImage(img_rbg)


