import cv2
import sys
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of CSRT, you can also use
    print(cv2.__version__)
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    
    tracker = cv2.TrackerMIL_create()

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
             tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        #elif tracker_type == "CSRT":
        #    tracker = cv2.TrackerCSRT_create()
 
    # Read video
    #video = cv2.VideoCapture("input.mp4")
    video = cv2.VideoCapture(-1) # for using CAM
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    
    ok, frame = video.read()
    
    if not ok:
        print ('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 183, 46, 42)
 
    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([170,50,50])
    upper = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    #res = cv2.GaussianBlur(result, (7, 7), 2)
    contours, hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = max(contours, key = len)
    bbox = cv2.boundingRect(contours)
    

    #bbox = cv2.selectROI(frame, False)
    #cnt = contours[0]
    #M = cv2.moments(cnt)
    #print( M )
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(res, bbox)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print("problem")
            break
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        kernal = np.ones((5, 5), "uint8")
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([170,50,50])
        upper = np.array([180,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        gray=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours = max(contours, key = len)
        bbox = cv2.boundingRect(contours)


        frame = res
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (25,155,155), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        #Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            break
    video.release()
    cv2.destroyAllWindows()