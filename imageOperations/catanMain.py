# import the opencv library
import cv2
import homography as hg
import colourThreshold as ct
import imgMorphologyOperations as imo
import argparse
import tileThreshold as tt
import featureMatchTiles as fmt
import time
# from adaptiveHistogramEqualisation import adaptiveHistEq

class BoardGrabber:
    def __init__(self, video_source, board_template):
        self.vid = video_source
        self.board_template = board_template
        self.cropCoords = None
        self.tileImage = None
        self.thiefImage = None
        self.numbersImage = None

        # will need current and previous board state here

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.M = None

    def getCroppedFrame(self, x, y, w, h):
        frame = self.getFrame()
        return frame[y:y+h, x:x+w]

    def getFrame(self):
        (ret, frame) = self.vid.read()
        if ret:
            return frame
        else:
            raise ValueError("Unable to read frame")
        
    
    def getHomographyTF(self):
        while True:
            frame = self.getFrame()

            dilated = imo.dilation(45, ct.getOceanThreshold(frame))
            x,y,w,h = imo.largestContourDetect(frame, dilated)

            cropped = frame[y:y+h, x:x+w]

            cv2.imshow("Cropped img", cropped)
            cv2.imshow("Dilated img", dilated)

            # Find the homography transform
            template = cv2.imread(self.board_template, 0)
            matchedPoints, flattened, self.M = hg.homographyTilt(cropped, template)

            if self.M is None:
                continue

            cv2.imshow("Warped Source Image", flattened)
            cv2.imshow("Matched points", matchedPoints)

            print("Is this a good homography? [y/n]")

            if cv2.waitKey(0) & 0xFF == ord('y'):
                self.tilesImage = flattened
                self.x = x
                self.y = y
                self.w = w
                self.h = h
                return
            elif cv2.waitKey(0) & 0xFF == ord('n'):
                self.M = None

            cv2.destroyAllWindows()


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument('img_dir', help='Path to testing images', default="catanImages/")
    parser.add_argument('video_index', help='Index of video to process', type=int, nargs='?', const=1)

    args = parser.parse_args()

    # Define a video capture object
    vid = cv2.VideoCapture(args.video_index)

    # Ensure camera is working
    if not vid.isOpened():
        print("Cannot open camera")
        exit()

    templateImage = f'{args.img_dir}/catanBoardTransparent2.png'

    board_grabber = BoardGrabber(vid, templateImage)
    board_grabber.getHomographyTF()

    cv2.waitKey(0)

    cv2.imwrite(f"{args.img_dir}/adjustedImg2.png", board_grabber.tilesImage)

    cv2.destroyAllWindows()

    # while(True):
        
    #     # Capture the video frame by frame
    #     ret, frame = vid.read()

    #     # Do stuff
    #     cropped_frame = frame[y:y+h, x:x+w]
    #     adjustedImage = cv2.warpPerspective(cropped_frame, M, (templateImage.shape[1],templateImage.shape[0]))

    #     # Display the resulting frame
    #     cv2.imshow('Adjusted Frame Live', adjustedImage)

    #     thresholdedImg = adjustedImage.copy()

    #     thresholder = tt.TileThresholder(thresholdedImg)

    #     for i, (x2, y2) in enumerate(thresholder):
    #         bb_size = 40
    #         currentTileImg = adjustedImage[y2-bb_size:y2+bb_size, x2-bb_size:x2+bb_size]
    #         cv2.circle(thresholdedImg, (x2, y2), 5, (0, 0, 255), -1)
    #         cv2.rectangle(
    #             thresholdedImg, (x2 - bb_size, y2 - bb_size), (x2 + bb_size, y2 + bb_size), (0, 255, 0), 2
    #         )

    #         #cv2.imshow("Current Tile", currentTileImg)
    #         thresholds = ct.getThresholds(currentTileImg)
    #         # for (k,v) in thresholds.items():
    #         #     print(f"Count for {k} is {cv2.countNonZero(v)}")

    #         # print out the key for the largest value size
    #         print(f"Tile {i} is {max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))}")
    #         cv2.putText(thresholdedImg, f"{max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))}", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #     cv2.imshow("image with labelled tiles", thresholdedImg)

    #     # the 'q' button is set as the quitting button you may use any
    #     # desired button of your choice
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    #     time.sleep(0.5)

    # # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()