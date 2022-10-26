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
    parser.add_argument("-d", '--dataset_dir', help="""Path to dataset to generate from""")   
    
    args = parser.parse_args()

    # Define a video capture object
    vid = dummyVid.dummyVideo(f"{args.dataset_dir}/{args.filename}")


    # Ensure camera is working
    if not vid.isOpened():
        print("Cannot open camera")
        exit()

    templateImage = f'{args.img_dir}/catanBoardTransparent2.png'

    board_grabber = BoardGrabber(vid, templateImage)
    board_grabber.getHomographyTF()
    # print(board_grabber.tilesImage)

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

    # for each tile image, call idNums.getCircularFeatures
    number_tiles = [idNums.getCircularFeatures(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)) for tile in tile_subimages]
    
    for i, tile in enumerate(number_tiles):
        # if tile is not None:
        cv2.imshow("Current tile", tile)
        # print(tile)
        cv2.waitKey(1000)
        result = input("What tile num is this?: ")
        print(f"{args.dataset_dir}{int(result)}/{i}_{args.filename}")
        # cv2.imwrite(args.dataset_dir)
        # print(result)


    # # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
