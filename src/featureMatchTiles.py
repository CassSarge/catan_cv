import cv2
import argparse

def loadTemplateImgs(img_dir):
    rockImg = cv2.imread(f'{img_dir}/rock.png', 0)
    fieldImg = cv2.imread(f'{img_dir}/field.png', 0)
    forestImg = cv2.imread(f'{img_dir}/forest.png', 0)
    wheatImg = cv2.imread(f'{img_dir}/wheat.png', 0)
    clayImg = cv2.imread(f'{img_dir}/clay.png', 0)
    desertImg = cv2.imread(f'{img_dir}/desert.png', 0)

    return rockImg, fieldImg, forestImg, wheatImg, clayImg, desertImg

def surfMatch(templateImg, targetImg):

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(templateImg,None)
    kp2, des2 = sift.detectAndCompute(targetImg,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    #matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    
    matchedPoints = cv2.drawMatches(templateImg,kp1,targetImg,kp2,good,None,**draw_params)

    cv2.imshow("matched points", matchedPoints)
    cv2.waitKey(0)

def checkAllTiles(rockImg, fieldImg, forestImg, wheatImg, clayImg, desertImg, targetImg):
    surfMatch(rockImg, targetImg)
    surfMatch(fieldImg, targetImg)
    surfMatch(forestImg, targetImg)
    surfMatch(wheatImg, targetImg)
    surfMatch(clayImg, targetImg)
    surfMatch(desertImg, targetImg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument('img_dir', help='Path to testing images')

    args = parser.parse_args()

    adjustedImage = cv2.imread(f'{args.img_dir}/adjustedImg.png')

    rockImg, fieldImg, forestImg, wheatImg, clayImg, desertImg = loadTemplateImgs(args.img_dir)

    surfMatch(fieldImg, adjustedImage)

