"""
https://github.com/iitmcvg/Content/blob/master/Sessions/Summer_School_2018/Session_1/Session_1.ipynb

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

def concatImgVerical(img):
    img=np.concatenate((img,img),axis=0)
    return img

def concatImgHorizontal(img):
    img=np.concatenate((img,img),axis=1)
    return img

def imgInfo(img):
    (row,col,cubeNum) = img.shape
    print("Number of Dimentions ={}".format(img.ndim))
    print("Height/Rows=", row)
    print("Width/Columns=", col)
    print("CubeNum/Channel=", cubeNum)
    print("Total Num of Pixel=", img.size)
    return (row,col,cubeNum)

def resize_img(img,scale_percent=50):
    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    """
    Preferable interpolation methods are cv2.INTER_AREA for shrinking and 
    cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR for zooming. By default, interpolation 
    method used is cv2.INTER_LINEAR for all resizing purposes. 
    """
    # resize image
    output = cv2.resize(img, dsize)
    return output

def splitChannels(img):
    (b, g, r) = cv2.split(img)
    return (b, g, r)

def mergeChannels(b, g, r):
    new_img = cv2.merge([b, g, r])
    return new_img

def cutImgLeft25per(img):
    rows=int(img.shape[0]/2)
    columns=int(img.shape[1]/2)
    cube_nums = img.shape[2]
    for cu in range(cube_nums):
        for col in range(0,columns):
            for row in range(rows,img.shape[0]):
                img[row,col,cu] = 0
    return img

def color2Grey(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def color2Hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def shift_image(img):
    """
     Translation is the shifting of object’s location. If you know the shift
     in (x,y) direction, let it be (t_x,t_y), you can create the transformation
     matrix M as follows:
     M =[1 0 t_x0 1 t_y]
     You can take make it into a Numpy array of type np.float32 and pass it into
     cv2.warpAffine() function. See below example for a shift of (100,50):
    """
    rows, cols, chan = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    RGB_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.title('My Image')
    plt.subplot(222)
    plt.imshow(RGB_dst)
    plt.title('Shifted Image')
    plt.show()



if __name__=="__main__":
    CurrentPath = os.getcwd()
    ImagePath = "\images\MessiOrignal.jpg"
    SourceImgPath = CurrentPath + ImagePath
    fileCheck(SourceImgPath)
    img = cv2.imread(SourceImgPath, 1)
    # dispImg(img,"default_pic")
    # conCatImg=concatImgVerical(img)
    # dispImg(conCatImg, "Concat_Vertical")
    #conCatImg=concatImgHorizontal(img)
    #dispImg(conCatImg, "Concat_Horizontal")
    #(row,col,cubeNum) = imgInfo(img)
    #half=resize_img(img,50)
    #dispImg(half,"half_pixels")
    # Double=resize_img(img,200)
    #dispImg(Double,"Double_pixels")
    # (b, g, r) = splitChannels(img)
    # new_img=mergeChannels(b, g, r)
    # dispImg(new_img, "merged_pic")
    # cut_img = cutImgLeft25per(img)
    # dispImg(cut_img, "cut_pic")
    # gray = color2Grey(img)
    # dispImg(gray, "gray_pic")
    # hsv = color2Hsv(img)
    # dispImg(hsv, "hsv_pic")
    shift_image(img)