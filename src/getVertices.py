import cv2
import tileThreshold as tt

def printMouseCoords(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x}, {y},")

def main():
    img = cv2.imread("catanImages/adjustedImg.png")
    cv2.imshow("img", img)

    cv2.setMouseCallback("img", printMouseCoords)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresholder = tt.TileThresholder(img, calibrate=False)
    print("tile centres are:")
    for (x,y) in thresholder:
        cv2.circle(img, (x,y), 5, (0,0,255), -1)
        print(f"{x}, {y},")

    print("vertices are:")
    for (x,y) in thresholder.vertices():
        cv2.circle(img, (x,y), 5, (0,0,255), -1)
        print(f"{x}, {y},")

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()
