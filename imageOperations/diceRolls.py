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
        dilatedImg = imo.dilation(20, img_threshold)
        x,y,w,h = imo.largestContourDetect(frame, dilatedImg)
        redDiceCropped = frame[y:y+h, x:x+w] # This image should be ballpark 200 pixels
        #cv2.imshow("Dilated img", dilatedImg)
        cv2.imshow("Dice red cropped", redDiceCropped)

        rDiceThresh = ct.getRedDiceThreshold(redDiceCropped)
        cv2.imshow("Dice red cropped+threshed", rDiceThresh)

        mask = cv2.bitwise_not(rDiceThresh) # invert
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 10, param1=30, param2=15, minRadius=6, maxRadius=30)


        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if ((len(circles) > 0) and (len(circles) <=6)): # no point guessing
                cv2.putText(mask,"RED: " + str(len(circles)), (5,5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
                #print(len(circles))
        else:
            print("brrr")

        if cv2.waitKey(1) == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()