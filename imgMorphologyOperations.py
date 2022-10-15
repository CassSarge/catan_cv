import cv2

def dilation(dilationSize, img):
    dilatation_size = dilationSize
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilated_img = cv2.dilate(img, element)
    return dilated_img

def largestContourCrop(img, thresh):
    # Find Canny edges
    edged = cv2.Canny(thresh, 30, 200)
    
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

    #cv2.drawContours(img, cnt, -1, (0, 255, 255), 10)

    #cv2.imshow('Largest Contour', img)
    #cv2.imshow('Canny Edges After Contouring', edged)

    #cv2.waitKey(0)

    x,y,w,h= cv2.boundingRect(sorted_contours[0])
        
    # cv2.waitKey(0)
    
    # print("Number of Contours found = " + str(len(contours)))

    return x,y,w,h