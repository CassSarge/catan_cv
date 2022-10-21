import cv2
from matplotlib.animation import ImageMagickBase

def dilation(dilationSize, img):
    dilatation_size = dilationSize
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilated_img = cv2.dilate(img, element)
    return dilated_img

def largestContourDetect(img, thresh):
    # Find Canny edges
    edged = cv2.Canny(thresh, 30, 200)
    
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

    #cv2.drawContours(img, cnt, -1, (0, 255, 255), 10)

    #cv2.imshow('Largest Contour', img)
    #cv2.imshow('Canny Edges After Contouring', edged)

    #cv2.waitKey(0)
    if len(sorted_contours) > 0:
        x,y,w,h= cv2.boundingRect(sorted_contours[0])
    else:
        x,y,w,h = 0,0,1920,1080
        print("Warning: len(sorted_contours < 0")
    # cv2.waitKey(0)

    return x,y,w,h

def NLargestContoursDetect(N, img, thresh, name):
    # Find Canny edges
    edged = cv2.Canny(thresh, 30, 200)
    
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    #print(sorted_contours[0])
    # cv2.drawContours(img, contours, -1, (0,255,0), 1)

    for (i,c) in enumerate(sorted_contours):
        if i < N:
            #print(i)
            x,y,w,h= cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 4)  
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        else:
            break

    return img