import numpy as np
import cv2
import argparse

def homographyTilt(tiltedImage, templateImage):
    
    MIN_MATCH_COUNT = 10
    #img1 = cv2.imread(tiltedImage,0 ) # Image with tilted perspective
    #img1colour = cv2.imread(tiltedImage)
    img1colour = tiltedImage
    img1 = cv2.cvtColor(img1colour, cv2.COLOR_BGR2GRAY)
    img2 = templateImage # Image to match to
    # img2colour = cv2.imread('catanImages/catanBoardTransparent.png') 

    #cv2.imshow("Template image", img2)
    #cv2.waitKey(0)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
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

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    
    matchedPoints = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    adjustedImage = cv2.warpPerspective(img1colour, M, (img2.shape[1],img2.shape[0]))

    #plt.imshow(matchedPoints),plt.show()

    return matchedPoints, adjustedImage, M


if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
    parser.add_argument('img_dir', help='Path to testing images')

    args = parser.parse_args()

    templateImage = cv2.imread(f'{args.img_dir}/catanBoardTransparent2.png', 0)
    tiltedImage = cv2.imread(f'{args.img_dir}/20221014_112636.jpg')
    #tiltedImage = 'catanImages/20221014_112641.jpg'

    
    matchedPoints, adjustedImage, M = homographyTilt(tiltedImage, templateImage)

    cv2.imshow("Warped Source Image", adjustedImage)
    cv2.imshow("Matched Points", matchedPoints)

    # Save image
    cv2.imwrite(f'{args.img_dir}/adjustedImg.png', adjustedImage)

    cv2.waitKey(0)