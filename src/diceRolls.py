import cv2
import colourThreshold as ct
import imgMorphologyOperations as imo
import argparse
import numpy as np

def cropToDie(img, colour):
    if colour == 'r':
        img_threshold = ct.getRedDiceThreshold(img, inlecture=False)
    elif colour == 'y':
        img_threshold = ct.getYellowDiceThreshold(img, inlecture=False)
    # cv2.imshow("Dice thresh", img_threshold)
    dilatedImg = imo.dilation(20, img_threshold)
    x,y,w,h = imo.largestContourDetect(frame, dilatedImg)
    dieCropped = frame[y:y+h, x:x+w] # This image should be ballpark 200 pixels
    #cv2.imshow("Dice red cropped", redDieCropped)
    return dieCropped

def getDieMask(img, colour):

    if colour == 'r':
        dieThresh = ct.getRedDiceThreshold(img)
    elif colour == 'y':
        dieThresh = ct.getYellowDiceThreshold(img)
    
    mask = cv2.bitwise_not(dieThresh) # invert
    mask = imo.erode(2, mask)
 
    return mask


def countPips(mask, dieCropped):
    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    numPips = 0
    output = dieCropped.copy()
    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        # if i == 0:
            # text = "examining component {}/{} (background)".format(
                # i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        # else:
            # text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        # print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        
        if area > 200 and area < 6000:
            # print("[INFO] area is {}".format(area))
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
            # componentMask = (labels == i).astype("uint8") * 255
            # show our output image and connected component mask
            #cv2.imshow("Connected Component", componentMask)
            numPips = numPips + 1
        else:
            #print("Skipped component")
            pass

   # print("[INFO] Dice roll result is {}".format(numPips))
    # cv2.imshow("Output", output)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    if numPips == 0:
        numPips = None

    return(numPips)


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Dice rolling with connected components')
    parser.add_argument("-v", '--video_index', help='Index of video to process', type=int, default=0)

    args = parser.parse_args()

    # Define a video capture object
    vid = cv2.VideoCapture(1)

    rNumList = [None]*15
    yNumList = [None]*15

    dice_in_hand = True

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
        cv2.waitKey(10)

        # Crop to the dice, get the mask for this colour
        dieCropped = cropToDie(frame, 'r')
        mask = getDieMask(dieCropped, 'r')
        # Count number of pips
        redNumPips = countPips(mask, dieCropped)
        # Put this number at the start of the list, remove last entry
        rNumList.insert(0, redNumPips)
        rNumList.pop()

        dieCropped = cropToDie(frame, 'y')        
        mask = getDieMask(dieCropped, 'y')
        yellowNumPips = countPips(mask, dieCropped)
        yNumList.insert(0, yellowNumPips)
        yNumList.pop()

        # Get the most commonly occuring number in the list (removes error results)
        if any(x is None for x in rNumList):
            redNumPipsMode = None
        else:
            redNumPipsMode = max(set(rNumList), key=rNumList.count)

        if any(x is None for x in yNumList):
            yellowNumPipsMode = None
        else:
            yellowNumPipsMode = max(set(yNumList), key=yNumList.count)

        
        if redNumPipsMode is None or yellowNumPipsMode is None:
            dice_in_hand = True
        else:
            if dice_in_hand == True:
                dice_in_hand = False
            # print(f"[Red] is {redNumPipsMode}, [Yellow] is {yellowNumPipsMode}")
                print(f"{redNumPipsMode + yellowNumPipsMode}")


    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()