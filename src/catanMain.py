# import the opencv library
import cv2
import homography as hg
import colourThreshold as ct
import imgMorphologyOperations as imo
import argparse
import tileThreshold as tt
import pixelCoords as pc
import featureMatchTiles as fmt
import identifyNumbers as idNums
from skimage.metrics import structural_similarity as compare_ssim
import time
# from adaptiveHistogramEqualisation import adaptiveHistEq

class Tile:
    def __init__(self, type, number, has_thief = False):
        self.type = type
        self.number = number
        self.has_thief = has_thief

    def __str__(self):
        return f"Tile: {self.type}, {self.number}, {self.has_thief}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.type == other.type and self.number == other.number and self.has_thief == other.has_thief

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.type, self.number, self.has_thief))

class BoardGrabber:
    def __init__(self, video_source, board_template):
        self.vid = video_source
        self.board_template = board_template
        self.cropCoords = None
        self.tileImage = None
        self.numbersImage = None
        self.thresholder = None

        self.thiefImage = None
        self.thiefTile = None

        self.hasbeenread = False
        # will need current and previous board state here

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.M = None

    def getCroppedFrame(self, x, y, w, h):
        frame = self.getFrame()
        return frame[y:y+h, x:x+w]

    def getFrame(self):
        # return cv2.imread("catanImages/screenshot10:16:54.png")
        (ret, frame) = self.vid.read()
        if ret:
            return frame
        else:
            raise ValueError("Unable to read frame")

    def getFlattenedFrame(self):
        img = self.getCroppedFrame(*self.cropCoords)
        template = cv2.imread(self.board_template, 0)

        adjustedImage = cv2.warpPerspective(img, self.M, (template.shape[1],template.shape[0]))
        return adjustedImage
    
    def getHomographyTF(self):
        while True:
            frame = self.getFrame()

            dilated = imo.dilation(20, ct.getOceanThreshold(frame))
            x,y,w,h = imo.largestContourDetect(frame, dilated)

            cropped = frame[y:y+h, x:x+w]

            cv2.imshow("Cropped img", cropped)
            cv2.imshow("Dilated img", dilated)

            # Find the homography transform
            template = cv2.imread(self.board_template, 0)
            matchedPoints, flattened, M = hg.homographyTilt(cropped, template)

            if M is None:
                continue

            cv2.imshow("Warped Source Image", flattened)
            cv2.imshow("Matched points", matchedPoints)

            print("Is this a good homography? [y/n]")

            if cv2.waitKey(0) & 0xFF == ord('y'):
                self.M = M
                self.tilesImage = flattened
                self.cropCoords = (x, y, w, h)
                self.thresholder = tt.TileThresholder(self.tilesImage)
                return
            elif cv2.waitKey(0) & 0xFF == ord('n'):
                self.M = None

    def getBoardState(self):
        curr = self.getFlattenedFrame()
        cv2.imshow('Adjusted Frame Live', curr)

        # use this to show information overlays
        curr_overlay = curr.copy()

        tiles = []
        centers = []

        for i, (x2, y2) in enumerate(self.thresholder):
            centers.append((x2, y2))

            bb_size = 40
            tile = curr[y2-bb_size:y2+bb_size, x2-bb_size:x2+bb_size]
            thresholds = ct.getTileThresholds(tile)

            cv2.circle(curr_overlay, (x2, y2), 5, (0, 0, 255), -1)
            cv2.rectangle(
                curr_overlay, (x2 - bb_size, y2 - bb_size), (x2 + bb_size, y2 + bb_size), (0, 255, 0), 2
            )
            cv2.putText(curr_overlay, f"{max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))}", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            most_likely_type = max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))
            most_likely_number = 0
            has_thief = i == self.thiefTile



            #cv2.imshow("Current Tile", currentTileImg)
            # for (k,v) in thresholds.items():
            #     print(f"Count for {k} is {cv2.countNonZero(v)}")

            # print out the key for the largest value size
            # print(f"Tile {i} is {most_likely_type}")

            tiles.append(Tile(most_likely_type, most_likely_number, has_thief))


        cv2.imshow("image with labelled tiles", curr_overlay)

        return (tiles, centers)
    
    def findThiefTile(self, verbose = False, expected_blobs = 1):
        curr = self.getFlattenedFrame()
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        base = self.tilesImage.copy()
        base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

        (percentage_changed, diff) = compare_ssim(base, curr, full=True)
        diff = (diff * 255).astype("uint8")

        thresholded_diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]

        # Show the differences between the images
        if verbose:
            print(f"Amount of pixels changed: {percentage_changed:.2f}%")
            cv2.imshow("Changed Regions", thresholded_diff)
        
        # dilate the image ad then find contours to find the thief
        dilated = imo.dilation(20, thresholded_diff)

        # find the largest contour 
        res = imo.NLargestContoursDetect(expected_blobs, curr, dilated, "Thief")

        # show the image with the contours in res plotted on top
        centroids = []
        for c in res:
            x,y,w,h= cv2.boundingRect(c)
            cv2.rectangle(base, (x, y), (x + w, y + h), (255,0,0), 4)  

            centroids.append((x + w//2, y + h//2))
    
        cv2.imshow("Thief", base)

        for (x,y) in centroids:
            closest_tile_index = min(enumerate(self.thresholder), key=lambda t: pc.PixelCoords.distPixels(t[1], pc.PixelCoords(x,y)))[0]
            if closest_tile_index != self.thiefTile:

                self.thiefTile = closest_tile_index
                # print(self.thiefTile)
                time.sleep(1)
                break
            else:
                print("I think the Thief hasnt moved")


        return centroids
    
    def checkForSettlements(self, verbose = False):
        curr = self.getFlattenedFrame()

        radius = 15
        threshRatio = 0.05

        for (x,y) in self.thresholder.vertices():
            cv2.circle(curr, (x,y), radius, (0,0,255), 1)

            vertex = curr[y-radius:y+radius, x-radius:x+radius]
            boxsize = radius*2

            
            thresholds = ct.getSettlementThresholds(vertex, inlecture = False)
            # cv2.imshow("White", thresholds["White"])
            # cv2.imshow("Orange", thresholds["Orange"])
            # cv2.imshow("Blue", thresholds["Blue"])
            # cv2.imshow("Red", thresholds["Red"])
            # cv2.waitKey(0)

            nonZeroMax = 0
            for key in thresholds:
                currentNonZero = cv2.countNonZero(thresholds[key])
                if currentNonZero > nonZeroMax:
                    nonZeroMax = currentNonZero
                

            if nonZeroMax > threshRatio*(boxsize*boxsize):
                cv2.putText(curr, f"{max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(curr, "None", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            


        cv2.imshow("Vertices", curr)

        return


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument("-d", '--img_dir', help='Path to testing images', default="catanImages/")
    parser.add_argument("-v", '--video_index', help='Index of video to process', type=int, default=0)

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
    # print(board_grabber.tilesImage)

    #cv2.waitKey(0)

    cv2.imwrite(f"{args.img_dir}/adjustedImg2.png", board_grabber.tilesImage)

    cv2.destroyAllWindows()

    while(True):
        board_grabber.findThiefTile()
        tiles, centers = board_grabber.getBoardState()
        board_grabber.checkForSettlements()
        # print(len(tiles))
        # print(tiles[0:3])
        # print(tiles[3:7])
        # print(tiles[7:12])
        # print(tiles[12:16])
        # print(tiles[16:19])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.2)
    

    # # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()