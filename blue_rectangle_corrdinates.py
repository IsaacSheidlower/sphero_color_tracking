import numpy as np
import cv2
import sys

if __name__ == '__main__' :
 
    video = cv2.VideoCapture(-1) # for using CAM

    ok, frame = video.read()
    
    if not ok:
        print ('Cannot read video file')
        sys.exit()
     
    print("AAAA")
    # Define an initial bounding box
    bbox = (287, 183, 46, 42)
 
    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60,110,100])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #cv2.imshow('mask',mask)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 5)
    #res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    shape = "unidentified"
    """peri = cv2.arcLength(4, True)
    approx = cv2.approxPolyDP(4, 0.04 * peri, True)

    if len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"""
    #ret,thresh = cv2.threshold(gray,0,0,0)
    im2,contours = cv2.findContours(gray, 1, 2)
    print(len(contours))
    bbox = cv2.boundingRect(im2[1])
    print(bbox)
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(res, p1, p2, (25,155,155), 2, 1)
    
    print(p1)
    print(p2)
    x_randCorr = np.random.randint(low=p1[0], high=p2[0])
    y_randCorr = np.random.randint(low=p1[1], high=p2[1])
    cv2.circle(res, (x_randCorr, y_randCorr), 20, (255,255, 255), thickness=-1)
    
    cv2.imshow("SSS", res)
    cv2.waitKey()