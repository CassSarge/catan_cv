import cv2


def adaptiveHistEq(bgr, gridsize):

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))

    lab[:,:,0] = clahe.apply(lab[:,:,0])

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgr


if __name__ == '__main__' :

    image_path = "catanImages/adjustedImg.png"
    gridsize = 16
    bgr = cv2.imread(image_path)
    cv2.imshow("Original image", bgr)

    bgr = adaptiveHistEq(bgr, gridsize)

    cv2.imshow("Adapted image", bgr)
    cv2.waitKey(0)
