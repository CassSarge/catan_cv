# import the opencv library
import cv2
import homography as hg
import colourThreshold as ct
import imgMorphologyOperations as imo


if __name__ == '__main__' :

    # Define a video capture object
    vid = cv2.VideoCapture(0)

    # Do setup stuff
    ret, frame = vid.read()

    # Get just the part of the frame that has the board in it
    dilatedImg = imo.dilation(10, ct.getOceanThreshold(frame))
    x,y,w,h = imo.largestContourDetect(frame, dilatedImg)
    contourCropped = frame[y:y+h, x:x+w]

    # Find the homography transform
    templateImage = cv2.imread('catanImages/catanBoardTransparent2.png', 0)
    matchedPoints, adjustedImage, M = hg.homographyTilt(contourCropped, templateImage)

    cv2.imshow("Warped Source Image", adjustedImage)
    cv2.imshow("Matched points", matchedPoints)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    while(True):
        
        # Capture the video frame by frame
        ret, frame = vid.read()

        # Do stuff
        cropped_frame = frame[y:y+h, x:x+w]
        adjustedImage = cv2.warpPerspective(cropped_frame, M, (templateImage.shape[1],templateImage.shape[0]))

        # Display the resulting frame
        cv2.imshow('Adjusted Frame Live', adjustedImage)

        forestThreshold = ct.getForestThreshold(adjustedImage)
        forestBoxes = imo.NLargestContoursDetect(4, adjustedImage, forestThreshold, "Forest")

        #fieldThreshold = ct.getFieldThreshold(adjustedImage)
        #fieldBoxes = imo.NLargestContoursDetect(4, forestBoxes, fieldThreshold, "Field")


        cv2.imshow("boxes", forestBoxes)
        cv2.imshow("threshold", forestThreshold)


        # the 'q' button is set as the quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()