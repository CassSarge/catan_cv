import numpy as np
import cv2
import argparse


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
                yield (x, y)

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
    pass

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

    print(PixelCoords(10, 6) // 3)

    img = cv2.imread(args.image)
    print(TileThresholder.list_iter.__next__())
    thresholder = TileThresholder(img, calibrate=args.calibrate)
    # thresholder = TileThresholder(img, calibrate=True)
    for (x, y) in thresholder:
        print((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        bb_size = 40
        cv2.rectangle(
            img, (x - bb_size, y - bb_size), (x + bb_size, y + bb_size), (0, 255, 0), 2
        )

    print("Showing image")

    cv2.imshow("image with labelled tiles", img)
    cv2.waitKey(5000)
