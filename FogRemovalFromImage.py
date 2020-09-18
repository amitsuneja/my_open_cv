import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
"""
https://www.sciencedirect.com/science/article/pii/S2215098616305067
https://github.com/iitmcvg/Content/blob/master/Sessions/Summer_School_2018/Session_2/HandsOn.ipynb
https://www.opencv-srf.com/2018/02/histogram-equalization.html
"""


def url2image(url):
  # download the image, convert it to a NumPy array, and then read
  # it into OpenCV format
  urlresponse = request.urlopen(url)
  image = np.asarray(bytearray(urlresponse.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # Channel interchange problem (RGB for matplotlib, and BGR for cv2)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def dispImg(img,windows_name="openCVPic"):
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.imshow(windows_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equalise_hist(img):
    """
    Now we do histogram equalisation -
    Before we estimate the airlight map, we will do histogram equalisation to increase the
    contrast of the image. This will give us a better estimate of the airlight map.
    What is Histogram equalization?
    It is a method in image processing of contrast adjustment using the image's
    histogram. Histogram equalization is one of the best methods for image
    enhancement. It provides better quality of images without loss of any
    information.
    The method is useful in images with backgrounds and foregrounds that are both bright
    or both dark.

    """
    # First convert the image to YUV format using cv2.cvtColor()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # Then equalise the Y channel of the image using cv2.equalizeHist()
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # Now convert the image back to BGR with cv2.cvtColor()
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # Now return the image
    return img_output


# We need to define an airlight map
def initialise_airmap(img, beta=0.9):
    img_c = img / 255  # Normalisation
    if img.shape[-1] > 1:  # If image has more than one channel
        min_img = beta * (np.amin(img_c, axis=2))  # Finds the minimum among all colour components of the image
    else:
        min_img = beta * img_c
    return min_img




if __name__=="__main__":
    img_url="http://brucedumes.me/wp-content/uploads/2012/12/foggy-ucla-1024x768.jpg"
    img = url2image(img_url)
    dispImg(img)
    new_img = equalise_hist(img)
    dispImg(new_img)
    new_img = initialise_airmap(new_img)
    plt.figure(figsize=(10, 16))
    plt.imshow(new_img, cmap='jet')
    plt.show()  # do not use dispImg here.



