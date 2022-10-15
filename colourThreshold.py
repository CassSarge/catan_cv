import numpy as np
import cv2

def openAndClose(img_threshold, ksize):

    kernel1 = np.ones((ksize,ksize),np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel1)
    kernel2 = np.ones((ksize*2,ksize*2),np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel2)

    return img_threshold

def getFieldThreshold(rgb_image):
    lower_field = np.array([0.115*179,0.357*255,0.429*255])
    upper_field = np.array([0.177*179,0.987*255,0.731*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_field, upper_field)

    img_threshold = openAndClose(img_threshold, 8)

    return img_threshold

def getOceanThreshold(rgb_image):

    lower_ocean = np.array([0.563*179,0.093*255,0.569*255])
    upper_ocean = np.array([0.619*179,0.772*255,0.902*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_ocean, upper_ocean)

    return img_threshold

def getForestThreshold(rgb_image):

    lower_forest = np.array([0.076*179,0.176*255,0.086*255])
    upper_forest = np.array([0.189*179,0.927*255,0.410*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_forest, upper_forest)

    img_threshold = openAndClose(img_threshold, 8)

    return img_threshold

if __name__ == '__main__' :

    img = cv2.imread("catanImages/adjustedImg.png")

    #cv2.imshow("Adjusted image", img)
    #cv2.waitKey(0)

    lower_forest = np.array([0.076*179,0.176*255,0.086*255])
    upper_forest = np.array([0.189*179,0.927*255,0.410*255])

    # lower_grass = np.array([0.129*179,0.489*255,0.294*255])
    # upper_grass = np.array([0.175*179,1.000*255,0.729*255])

    window_name = "Adjusted Image"

    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_forest, upper_forest)

    cv2.imshow(window_name, img)
    cv2.imshow(window_name, img_threshold)
    cv2.waitKey(0)

    img_threshold = getOceanThreshold(img)
    #dilation(10, img_threshold)

