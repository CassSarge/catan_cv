import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt


class PixelCoords:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return PixelCoords(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return PixelCoords(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return PixelCoords(self.x * other, self.y * other)

    def __floordiv__(self, other):
        return PixelCoords(self.x // other, self.y // other)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __iter__(self):
        return iter((self.x, self.y))
    
    def distPixels(first, second):
        # print(first)
        # print(second)
        return (first.x - second.x) ** 2 + (first.y - second.y) ** 2



class TileThresholder:
    def __init__(self, image, calibrate=False):
        self.delta_r = PixelCoords(0, 0)
        self.delta_dr = PixelCoords(0, 0)
        self.delta_dl = PixelCoords(0, 0)

        if calibrate:
            self.top_three_tiles = self.calibrateTileCoords(image)
            self.origin = self.top_three_tiles[0]
            self.calibrate()
        else:
            self.autocalibrate()
        print(
            f"Origin: {self.origin}, Right: {self.delta_r}, Down Right: {self.delta_dr}, Down Left: {self.delta_dl}"
        )

    def calibrate(self):
        self.delta_r = (self.top_three_tiles[1] - self.top_three_tiles[0]) // 2
        self.delta_dr = (self.top_three_tiles[2] - self.top_three_tiles[0]) // 4
        self.delta_dl = self.delta_dr - self.delta_r

    def autocalibrate(self):
        self.origin = PixelCoords(257, 153)
        self.delta_r = PixelCoords(114, -1)
        self.delta_dr = PixelCoords(63, 95)
        self.delta_dl = self.delta_dr - self.delta_r

    def calibrateTileCoords(self, img):
        cv2.imshow("image", img)
        top_tiles_positions = []
        cv2.setMouseCallback("image", printMouseCoords, top_tiles_positions)
        while True:
            # wait for key press
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        return top_tiles_positions

    def getTileCoords(self, row, col):
        pos = self.origin
        pos += self.delta_r * col
        for i in range(row):
            if i < 2:
                pos += self.delta_dl
            else:
                pos += self.delta_dr
        return pos

    def __iter__(self):
        row_lengths = [3, 4, 5, 4, 3]
        for i, rng in enumerate(row_lengths):
            for j in range(rng):
                (x, y) = self.getTileCoords(i, j)
                yield PixelCoords(x, y)

    listOfKeyPoints = [
        "Centre of top row leftmost tile",
        "Centre of top row 4rd tile",
        "Centre of last row final tile",
        "Centre of last row first tile" "Please Press Esc",
    ]

    list_iter = iter(listOfKeyPoints)


def printMouseCoords(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        param.append(PixelCoords(x, y))
        print(TileThresholder.list_iter.__next__())

def getBoundingBox(img: np.ndarray, coords: PixelCoords, size: int):
    # returns the bounding box of size size at the tile at coords in img as a numpy array
    (x,y) = coords
    print((x,y))
    bb = img[y-size//2:y+size//2, x-size//2:x+size//2, :]
    return bb

def getMetric(img: np.ndarray, coords: PixelCoords, size: int):
    # returns the histogram of the bounding box of size size at the tile at coords in img
    # get bounding box
    bb = getBoundingBox(img, coords, size)
    cv2.imshow("bounding box", bb)
    # cv2.waitKey(0)
    # # calc avg hue 
    # hue_avg = bb.mean(axis=(0,1))

    # calc histogram in HSV space
    hist = cv2.calcHist([bb], [0], None, [179], [0, 179])

    return hist

def autoThreshold(tt: TileThresholder, img: np.ndarray, size: int):
    # returns a numpy array of the same shape as img with the thresholded values
    metrics = np.array([getMetric(img, coords, size) for coords in tt])
    print(len(metrics))
    print(metrics)
    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculates the thresholds for each type of tile based on a given image from above the Catan board"
    )
    parser.add_argument("image", help="Path to the image")
    parser.add_argument(
        "-c",
        "--calibrate",
        action="store_true",
        help="Calibrate the position of the tiles, based on user input",
    )
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print(TileThresholder.list_iter.__next__())
    thresholder = TileThresholder(img_bgr, calibrate=args.calibrate)
    # thresholder = TileThresholder(img, calibrate=True)
    print("Showing image")

    hue_hists = autoThreshold(thresholder, img, 80)

    label = [0,1,2,1,3,2,4,3,2,4,3,1,0,5,0,2,4,1,3] 
    # # for each histogram in hue_hists, plot it with a title corresponding to the value in the label list
    # for i in [4, 7, 10, 18]: WHEAT
    # for i in [0, 12, 14]: ROCK
    for i in [3, 5, 8, 15]:
        plt.plot(hue_hists[i])
        plt.title(str(label[i]))
        plt.xlim([0,179])
        plt.show()

    # calculate the histogram of img and plot it
    # hist = cv2.calcHist([img], [0], None, [179], [0, 60])
    # plt.plot(hist)
    # plt.xlim([0,179]) 

    plt.show()

    for (x, y) in thresholder:
        print((x, y))
        cv2.circle(img_bgr, (x, y), 5, (0, 0, 255), -1)
        bb_size = 40
        cv2.rectangle(
            img, (x - bb_size, y - bb_size), (x + bb_size, y + bb_size), (0, 255, 0), 2
        )


    





    # plt.plot(hue_hists[0], color='r')
    
    
    
    
    # print(hue_averages)
    # hue = hue_averages[:,0]
    # sat = hue_averages[:,1]
    # val = hue_averages[:,2]
    # label = [0,1,2,1,3,2,4,3,2,4,3,1,0,5,0,2,4,1,3]
    # colors = ["rock", "field", "forest", "wheat", "clay", "desert"]


    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(hue, sat, val, marker='o', c=label)

    # ax.set_xlabel('B')
    # ax.set_ylabel('G')
    # ax.set_zlabel('R')

    cv2.imshow("image with labelled tiles", img)
    cv2.waitKey(5000)
