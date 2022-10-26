import cv2
import imgMorphologyOperations as imo
from catanMain import BoardGrabber
import argparse
import tileThreshold as tt
import featureMatchTiles as fmt
import identifyNumbers as idNums
import dummyVideo as dummyVid

if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description="""File to grab number tiles
            from a board image and prepare them for classification""")
    parser.add_argument("-i", "--img_dir", help="Path to the image directory", default="../catanImages/")
    parser.add_argument("-f", '--filename', help="""Path to image to generate
    dataset from""")
    
    args = parser.parse_args()

    # Define a video capture object
    vid = dummyVid.dummyVideo(args.filename)


    # Ensure camera is working
    if not vid.isOpened():
        print("Cannot open camera")
        exit()

    templateImage = f'{args.img_dir}/catanBoardTransparent2.png'

    board_grabber = BoardGrabber(vid, templateImage)
    board_grabber.getHomographyTF()
    # print(board_grabber.tilesImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    board_grabber.findThiefTile()
    tiles, centers = board_grabber.getBoardState()

    bb_size = 45

    image = board_grabber.getFlattenedFrame()

    # for each center, grab the bounding box aronud the center and combine them into a list
    tile_subimages = []
    number_subimages = []
    for i, (x,y) in enumerate(centers):
        tile_subimages.append(image[y-bb_size:y+bb_size, x-bb_size:x+bb_size])
        number_cropped = idNums.getCircularFeatures(cv2.cvtColor(tile_subimages[i], cv2.COLOR_BGR2GRAY))



    # for each tile image, call idNums.getCircularFeatures
    number_tiles = [idNums.getCircularFeatures(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)) for tile in tile_subimages]

    #convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
