import cv2
import colourThreshold as ct
import imgMorphologyOperations as imo
from datetime import datetime
import argparse


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument('img_dir', help='Path to testing images')

    args = parser.parse_args()

    # Define a video capture object
    vid = cv2.VideoCapture(2)

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
        dilatedImg = imo.dilation(20, ct.getOceanThreshold(frame))
        x,y,w,h = imo.largestContourDetect(frame, dilatedImg)
        contourCropped = frame[y:y+h, x:x+w]
        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Cropped Feed', dilatedImg)
        
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('s'):
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            img_name = f"{args.img_dir}/screenshot" + current_time + ".png"
            cv2.imwrite(img_name, frame)
            print('Screenshot ', img_name, ' saved')

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
