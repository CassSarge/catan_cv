import cv2
import colourThreshold as ct
import imgMorphologyOperations as imo
import argparse
import numpy as np

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument('img_dir', help='Path to testing images')

    args = parser.parse_args()

    # Define a video capture object
    vid = cv2.VideoCapture(1)

    # Ensure camera is working
    if not vid.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = vid.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # Get just the part of the frame that has the board in it
        cv2.imshow("Dice", frame)

        img_threshold = ct.getRedDiceThreshold(frame)
        #cv2.imshow("Dice red thresh", img_threshold)
        dilatedImg = imo.dilation(10, img_threshold)
        x,y,w,h = imo.largestContourDetect(frame, dilatedImg)
        redDiceCropped = frame[y:y+h, x:x+w] # This image should be ballpark 200 pixels
        #cv2.imshow("Dilated img", dilatedImg)

        redDiceCropped = cv2.medianBlur(redDiceCropped,5)

        cv2.imshow("Dice red cropped", redDiceCropped)

        rDiceThresh = ct.getRedDiceThreshold(redDiceCropped)

        rDiceThresh = imo.erode(5, rDiceThresh)

        # mask = cv2.bitwise_not(rDiceThresh) # invert
        
        # apply connected component analysis to the thresholded image
        output = cv2.connectedComponentsWithStats(rDiceThresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # loop over the number of unique connected component labels
        for i in range(0, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format( i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            # clone our original image (so we can draw on it) and then draw
            # a bounding box surrounding the connected component along with
            # a circle corresponding to the centroid
            output = redDiceCropped.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID
            componentMask = (labels == i).astype("uint8") * 255
            # show our output image and connected component mask
            cv2.imshow("Output", output)
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()