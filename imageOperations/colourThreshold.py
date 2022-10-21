import numpy as np
import cv2
import argparse

def closeAndOpen(img_threshold, ksize):

    kernel1 = np.ones((ksize,ksize),np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel1)

    kernel2 = np.ones((int(ksize*1.8),int(ksize*1.8)),np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel2)

    return img_threshold

def getWheatThreshold(rgb_image, inlecture=False):
    # PNR Threshold
    lower_wheat = np.array([0.089*179,0.707*255,0.386*255])
    upper_wheat = np.array([0.132*179,1.000*255,0.720*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_wheat, upper_wheat)

    img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold

def getRockThreshold(rgb_image, inlecture=False):
    # PNR Threshold
    lower_rock = np.array([0.929*179,0.000*255,0.000*255])
    upper_rock = np.array([0.099*179,1.000*255,1.000*255])

    # ????? fucked

    # # % Define thresholds for channel 1 based on histogram settings
    # channel1Min = 0.875;
    # channel1Max = 0.098;

    # % Define thresholds for channel 2 based on histogram settings
    # channel2Min = 0.304;
    # channel2Max = 0.584;

    # % Define thresholds for channel 3 based on histogram settings
    # channel3Min = 0.218;
    # channel3Max = 0.526;


    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_rock, upper_rock)

    #img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold

def getFieldThreshold(rgb_image, inlecture=False):
    # PNR Threshold
    lower_field = np.array([0.115*179,0.357*255,0.429*255])
    upper_field = np.array([0.177*179,0.987*255,0.731*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_field, upper_field)

    img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold

def getClayThreshold(rgb_image, inlecture=False):
    # PNR Threshold
    lower_clay = np.array([0.045*179,0.580*255,0.121*255])
    upper_clay = np.array([0.083*179,1.00*255,0.616*255])

    # % Define thresholds for channel 1 based on histogram settings
    # channel1Min = 0.045;
    # channel1Max = 0.083;

    # % Define thresholds for channel 2 based on histogram settings
    # channel2Min = 0.580;
    # channel2Max = 1.000;

    # % Define thresholds for channel 3 based on histogram settings
    # channel3Min = 0.121;
    # channel3Max = 0.616;

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_clay, upper_clay)

    img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold


def getOceanThreshold(rgb_image, inlecture=False):

    # lower_ocean = np.array([0.563*179,0.093*255,0.569*255])
    # upper_ocean = np.array([0.619*179,0.772*255,0.902*255])

    # lower_ocean = np.array([0.572*179,0.000*255,0.512*255])
    # upper_ocean = np.array([0.732*179,0.840*255,0.755*255])

    lower_ocean = np.array([0.540*179,0.000*255,0.325*255])
    upper_ocean = np.array([0.650*179,0.987*255,0.961*255]) 

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_ocean, upper_ocean)

    img_threshold = closeAndOpen(img_threshold, 10)

    return img_threshold

def getForestThreshold(rgb_image, inlecture=False):

    # Lecture theatre threshold
    if inlecture:
        lower_forest = np.array([0.145*179,0.125*255,0.235*255])
        upper_forest = np.array([0.227*179,0.573*255,0.673*255])
    # PNR threshold 
    else:
        lower_forest = np.array([0.108*179,0.274*255,0.000*255])
        upper_forest = np.array([0.146*179,0.957*255,0.604*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(frame_HSV, lower_forest, upper_forest)

    img_threshold = closeAndOpen(img_threshold, 10)

    return img_threshold

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument('img_dir', help='Path to testing images')

    args = parser.parse_args()

    img = cv2.imread(f"{args.img_dir}/adjustedImg.png")

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

