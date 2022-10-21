import numpy as np
import cv2
import argparse

def getThresholds(rgb_image, inlecture = False):
    img_wheat = getWheatThreshold(rgb_image, inlecture)
    img_rock = getRockThreshold(rgb_image, inlecture)
    img_field = getFieldThreshold(rgb_image, inlecture)
    img_clay = getClayThreshold(rgb_image, inlecture)
    img_forest = getForestThreshold(rgb_image, inlecture)
    # img_desert = getDesertThreshold(rgb_image, inlecture)

    results = {"wheat": img_wheat, "rock": img_rock, "field": img_field, "clay":
            img_clay, "forest": img_forest}
    
    return results 

def inRangeWrapper(img, lower, upper):
    if lower[0] > upper[0]:
        img1 = cv2.inRange(img, lower, np.array([179, upper[1], upper[2]]))
        img2 = cv2.inRange(img, np.array([0, lower[1], lower[2]]), upper)
        img = cv2.bitwise_or(img1, img2)
    else:
        img = cv2.inRange(img, lower, upper)

    return img 

def closeAndOpen(img_threshold, ksize):

    kernel1 = np.ones((ksize,ksize),np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel1)

    kernel2 = np.ones((int(ksize*1.8),int(ksize*1.8)),np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel2)

    return img_threshold

def getWheatThreshold(rgb_image, inlecture=False):
    
    if inlecture:
        # Lecture theatre threshold
        lower_wheat = np.array([0.125*179,0.570*255,0.547*255])
        upper_wheat = np.array([0.151*179,1.000*255,1.000*255])
    else:
        # PNR Threshold
        lower_wheat = np.array([0.089*179,0.707*255,0.386*255])
        upper_wheat = np.array([0.132*179,1.000*255,0.720*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = inRangeWrapper(frame_HSV, lower_wheat, upper_wheat)

    img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold

def getRockThreshold(rgb_image, inlecture=False):

    if inlecture:
        # Lecture theatre Threshold
        lower_rock = np.array([0.667*179,0.000*255,0.392*255])
        upper_rock = np.array([0.149*179,0.415*255,0.941*255])
    else:
        # PNR Threshold
        lower_rock = np.array([0.875*179,0.304*255,0.218*255])
        upper_rock = np.array([0.098*179,0.584*255,0.526*255])


    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = inRangeWrapper(frame_HSV, lower_rock, upper_rock)

    #img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold

def getFieldThreshold(rgb_image, inlecture=False):

    if inlecture:
        # Lecture theatre Threshold
        lower_field = np.array([0.169*179,0.120*255,0.592*255])
        upper_field = np.array([0.287*179,0.865*255,0.863*255])
    else:
        # PNR Threshold
        lower_field = np.array([0.115*179,0.357*255,0.429*255])
        upper_field = np.array([0.177*179,0.987*255,0.731*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = inRangeWrapper(frame_HSV, lower_field, upper_field)

    img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold

def getClayThreshold(rgb_image, inlecture=False):
    
    if inlecture:
        # Lecture theatre
        lower_clay = np.array([0.085*179,0.409*255,0.268*255])
        upper_clay = np.array([0.128*179,1.00*255,1.000*255])
    else:
        # PNR Threshold
        lower_clay = np.array([0.045*179,0.580*255,0.121*255])
        upper_clay = np.array([0.083*179,1.00*255,0.616*255])

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = inRangeWrapper(frame_HSV, lower_clay, upper_clay)

    img_threshold = closeAndOpen(img_threshold, 8)

    return img_threshold


def getOceanThreshold(rgb_image, inlecture=False):

    if inlecture:
        lower_ocean = np.array([0.548*179,0.103*255,0.599*255])
        upper_ocean = np.array([0.637*179,0.987*255,0.992*255]) 
    else:
        lower_ocean = np.array([0.540*179,0.000*255,0.325*255])
        upper_ocean = np.array([0.650*179,0.987*255,0.961*255]) 

    frame_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    img_threshold = inRangeWrapper(frame_HSV, lower_ocean, upper_ocean)

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
    img_threshold = inRangeWrapper(frame_HSV, lower_forest, upper_forest)

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
    img_threshold = inRangeWrapper(frame_HSV, lower_forest, upper_forest)

    cv2.imshow(window_name, img)
    cv2.imshow(window_name, img_threshold)
    cv2.waitKey(0)

    img_threshold = getOceanThreshold(img)
    #dilation(10, img_threshold)

