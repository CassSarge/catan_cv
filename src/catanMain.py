# import the opencv library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import homography as hg
import colourThreshold as ct
import imgMorphologyOperations as imo
import argparse
import tileThreshold as tt
import pixelCoords as pc
import dummyVideo as dummyVid
from skimage.metrics import structural_similarity as compare_ssim
import time
import identifyNumbers as idNums
import predict as pd
from collections import defaultdict

class Vertex:
    def __init__(self, x, y, settlement_colour = None):
        self.settlement_colour = settlement_colour
        self.coords = pc.PixelCoords(x, y)
    
class Tile:
    def __init__(self, type, number, tile_image, has_thief = False):
        self.type = type
        self.tile_image = tile_image
        self.number = number
        self.has_thief = has_thief
        self.number = None
        self.vertices = None

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
    def __init__(self, video_source, board_template, inlecture):
        self.vid = video_source
        self.board_template = board_template
        self.cropCoords = None
        self.tileImage = None
        self.numbersImage = None
        self.thresholder = None
        self.inlecture = inlecture

        self.thiefImage = None
        self.thiefTile = None

        self.Tiles = None
        self.Centres = None
        self.Vertices = None

        self.hasbeenread = False

        self.lastDiceRoll = None

        self.previousImage = None
        self.latestImage = None
        # will need current and previous board state here

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.M = None

    def getCroppedFrame(self, x, y, w, h):
        frame = self.getFrame()
        return frame[y:y+h, x:x+w]

    def getFrame(self):
        for _ in range(4):
            self.vid.grab()
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

            dilated = imo.dilation(20, ct.getOceanThreshold(frame, self.inlecture))
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
        centres = []

        for i, (x2, y2) in enumerate(self.thresholder):
            centres.append(pc.PixelCoords(x2, y2))

            bb_size = 40
            tile_img = curr[y2-bb_size:y2+bb_size, x2-bb_size:x2+bb_size]
            thresholds = ct.getTileThresholds(tile_img, self.inlecture)

            cv2.circle(curr_overlay, (x2, y2), 5, (0, 0, 255), -1)
            cv2.rectangle(
                curr_overlay, (x2 - bb_size, y2 - bb_size), (x2 + bb_size, y2 + bb_size), (0, 255, 0), 2
            )
            cv2.putText(curr_overlay, f"{max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))}", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            most_likely_type = max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))
            most_likely_number = 0
            has_thief = i == self.thiefTile

            tiles.append(Tile(most_likely_type, most_likely_number, tile_img, has_thief))

        self.Tiles = tiles
        self.Centres = centres

        # cv2.imshow("image with labelled tiles", curr_overlay)

        return curr_overlay
    
    def updateLatestFrame(self):
        self.previousImage = self.latestImage
        self.latestImage = self.getFlattenedFrame()
        if self.previousImage is None:
            self.previousImage = self.latestImage

        return

    def findThiefTile(self, verbose = False, expected_blobs = 1):
        curr = self.latestImage.copy()
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

        radius = 15 # 10
        threshRatio = 0.25 # 0.2 ?

        vertices = []
        for (x,y) in self.thresholder.vertices():

            vertex = curr[y-radius:y+radius, x-radius:x+radius]
            boxsize = radius*2

            
            thresholds = ct.getSettlementThresholds(vertex, self.inlecture)
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
                cv2.circle(curr, (x,y), radius, (0,0,255), 1)
                vertices.append(Vertex(x, y, settlement_colour = f"{max(thresholds, key=lambda k: cv2.countNonZero(thresholds[k]))}"))
            else:
                # cv2.putText(curr, "None", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                vertices.append(Vertex(x, y, settlement_colour = None))

        self.Vertices = vertices
        cv2.imshow("Vertices", curr)

        return

    def getRelatedSettlements(self):
    
        tile_radius = 100 ** 2
        for i, (x,y) in enumerate(self.Centres):
            self.Tiles[i].vertices = []
            vertex_distances = list(map(lambda t: pc.PixelCoords.distPixels(t.coords, pc.PixelCoords(x,y)), self.Vertices))
            for j, dist in enumerate(vertex_distances):
                if dist < tile_radius:
                    self.Tiles[i].vertices.append(self.Vertices[j])
                    # Woohoo this vertex is with this tile

    def classifyNumbers(self):
        # for tile in self.Tiles:
        #     cv2.imshow("Tile", tile.tile_image)
        #     cv2.waitKey(0)

        m = pd.load_model("model/epochs1000.hdf5")

        bb_size = 45

        image = board_grabber.getFlattenedFrame()

        # for each center, grab the bounding box aronud the center and combine them into a list
        tile_subimages = []
        for i, (x,y) in enumerate(self.Centres):
            tile_subimages.append(image[y-bb_size:y+bb_size, x-bb_size:x+bb_size])

        # for each tile image, call idNums.getCircularFeatures
        number_tiles = [idNums.getCircularFeatures(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)) for tile in tile_subimages]
        

        #make copy of blank board
        numberedBlank = self.tilesImage.copy()
        

        for i, tile in enumerate(number_tiles):
            if tile is not None:
                #cv2.imshow("Current tile", tile)
                self.Tiles[i].number = pd.predictNumberFromImg(tile, m)
                numberedBlank = cv2.puttext(numberedBlank, self.Tiles[i].number, (self.Centers[i][0],self.Centers[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
            else:
                self.Tiles[i].has_thief = True
                self.Tiles[i].type = "desert"

        cv2.imshow(numberedBlank)
                

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument("-d", '--img_dir', help='Path to testing images', default="catanImages/")
    parser.add_argument("-v", '--video_index', help='Index of video to process', type=int, default=0)
    parser.add_argument("-f", "--filename", help="Full filepath to use instead of video stream", default=None)
    parser.add_argument("-l", "--location", help="Location for colour thresholding, 'pnr' or 'lecture'", default="pnr")

    args = parser.parse_args()

    # Define a video capture object
    if args.filename is None:
        vid = cv2.VideoCapture(args.video_index)
    # If we are using a dummy video (still image) instead
    else:
        vid = dummyVid.dummyVideo(f"{args.filename}")

    # Decide which set of lighting conditions to use for colour masks
    if args.location == "pnr":
        inlecture = False
    elif args.location == "lecture":
        inlecture = True

    # Ensure camera is working
    if not vid.isOpened():
        print("Cannot open camera")
        exit()

    # Template image to perform homography to
    templateImage = f'{args.img_dir}/catanBoardTransparent2.png'

    board_grabber = BoardGrabber(vid, templateImage, inlecture)


    # Get a good homography
    board_grabber.getHomographyTF()

    cv2.imwrite(f"{args.img_dir}/adjustedImg2.png", board_grabber.tilesImage)

    cv2.destroyAllWindows()

    # Get board state and wait for it to show
    tile_overlay = board_grabber.getBoardState()
    cv2.imshow("Tile overlay", tile_overlay)
    cv2.waitKey(1000)

    print("Please place the number tiles on each hexagon and the thief")
    input("Press enter when ready: ")

    # Identify numbers and add it to the board state, along with an overlay for it
    # Store all the numbers in the tiles objects
    board_grabber.classifyNumbers()

    # Wait for user input to say everyone has placed their settlements and roads
    print("Please have each player place their first two settlements and roads")
    input("Press enter when ready: ")
    # Start main loop
    board_grabber.updateLatestFrame()
    board_grabber.updateLatestFrame()
    

    # Set previous frame as current frame

    # on a dice roll
    while(True):
        diceroll = input("Enter dice roll: ")

        if diceroll == 7:
            # Get and show latest frame
            board_grabber.updateLatestFrame()
            input("You have rolled a 7! Please move the thief to another Tile and press Enter!")
            board_grabber.updateLatestFrame()

            # Check if thief has moved from previous round by comparing with previous frame
            board_grabber.findThiefTile()

        # Check where settlements are
        board_grabber.checkForSettlements()
        board_grabber.getRelatedSettlements()

        # for i, tile in enumerate(board_grabber.Tiles):
        #     print(i)
        #     for vertex in tile.vertices:
        #         print(vertex.settlement_colour)
        #     print("----")
        

        # Loop through settlements and check what each player should get based on latest 
        board_grabber.lastDiceRoll = 6

        playerColours = ["Blue", "Red", "Orange", "White"]
        playerUpdates = {key : defaultdict(int) for key in playerColours}

        terminal_colours = {"Blue" : '\033[94m', "Orange" : '\033[93m', "Red" : '\033[91m', "White" : '\033[0m'}

        for tile in board_grabber.Tiles:
            # print(f"{tile.number=} vs {board_grabber.lastDiceRoll=}")
            if tile.number == board_grabber.lastDiceRoll:
                for vertex in tile.vertices:
                    if vertex.settlement_colour is not None:
                        playerUpdates[vertex.settlement_colour][tile.resource] += 1
                        # print(f"{vertex.settlement_colour} pick up a {tile.type}")

        for colour in playerUpdates:
            print(f"{terminal_colours[colour]}{colour} gets ", end="")
            all_pickups = list(map(lambda x: f"{x[1]} {x[0]}", playerUpdates[colour].items()))
            print(" and ".join(all_pickups))
            print(terminal_colours["White"])

        
        # dice roll result



        # print(len(tiles))
        # print(tiles[0:3])
        # print(tiles[3:7])
        # print(tiles[7:12])
        # print(tiles[12:16])
        # print(tiles[16:19])
        
        # Wait for user input saying its the next turn
            # Later wait for a new dice roll and loop again after that
        print("_______")
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print("Would anybody like to play a knight card?")
        if input("Enter y/n: ") == "y":
            board_grabber.updateLatestFrame()
            print("Please move the thief to another Tile and press Enter!")
            board_grabber.updateLatestFrame()
            board_grabber.findThiefTile()

        time.sleep(0.2)
    
    # # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()