# import the opencv library
import cv2
import argparse
import numpy as np


def getCircularFeatures(im_base):

    im = im_base.copy()

    #expectation of 18 number tiles placed in board
    # im = cv2.equalizeHist(im)
    im_blur = im.copy()
    # im_blur = cv2.GaussianBlur(im, (3,3), cv2.BORDER_DEFAULT)

    # print(im_blur.shape)

    #detect only one circle
    circular_features = cv2.HoughCircles(im_blur, cv2.HOUGH_GRADIENT, 1, 10, param1=60, param2=30, minRadius=15, maxRadius=25)
    if (circular_features is None):
        print("No circles found")
        return None
    # print("here")
    # print(circular_features)
    circular_features = np.around(circular_features).astype("uint16")


    # print(circular_features[:,:,:])

    for i in circular_features[0,:]:
        # print(i)
        
        # # draw the outer circle
        # cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)
        # # draw the center of the circle
        # cv2.circle(im,(i[0],i[1]),2,(0,0,255),3)

        # # if pixel is outside of the circle, set it to 0
        # for x in range(im.shape[0]):
        #     for y in range(im.shape[1]):
        #         if (x-i[0])**2 + (y-i[1])**2 > i[2]**2:
        #             im[x,y] = 0

        shrink_radii = 1
        im_mask = np.zeros(im.shape, dtype=np.uint8)
        cv2.circle(im_mask, (i[0],i[1]), i[2]-shrink_radii, (255,255,255),-1)
        masked_image = cv2.bitwise_and(im_mask,im)

        # crop the image to the circle
        cropped_im = masked_image[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

    
    

    

    # cv2.imshow('detected circles',cropped_im)
    # cv2.waitKey(0)



    return cropped_im



def getCircular(im):

    #increase image contrast
    #im = cv2.equalizeHist(im)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 30

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    #params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    params.filterByInertia = False
    #params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    keypoints = detector.detect(im)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(im, keypoints, blank, (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return 0








if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Detecting Circular Numbers')
    parser.add_argument('img_dir', help='Path to testing images')

    args = parser.parse_args()

    im = cv2.imread(f'{args.img_dir}tile_number_3.jpg', 0)

    #images taken from phone too large DELETE AFTER TESTING##########
    # x = int(im.shape[1] * 20 / 100)
    # y = int(im.shape[0] * 20 / 100)
    # dim = (x,y)
    # im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    #################################################################


    getCircularFeatures(im)

    # cv2.imshow('detected circles',im)
    # cv2.waitKey(0)
