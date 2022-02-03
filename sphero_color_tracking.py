import cv2
import sys
import numpy as np
import rospy 
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Point, Quaternion

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 

class TrackSphero():    
    def __init__(self):
        self.poi_pub = rospy.Publisher("/aabl/poi", PoseStamped, queue_size=10)
        self.tracker_type = cv2.TrackerMIL_create()
        self.tracker = cv2.TrackerMIL_create()
        # Read video
        #video = cv2.VideoCapture("input.mp4")
        self.video = cv2.VideoCapture(-1) # for using CAM
        self.lower_color_bound = np.array([60,110,100])
        self.upper_color_bound = np.array([130,255,255])
    
    # Set up tracker.
    # Instead of CSRT, you can also use
    """print(cv2.__version__)
 
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
        #    tracker = cv2.TrackerCSRT_create()"""
 
    def track_sphero(self):
 
        # Exit if video not opened.
        while not self.video.isOpened():
            print("Could not open video")
    
        # Read first frame.
        
        ok, frame = self.video.read()
        
        while not ok:
            print ('Cannot read video file')
            
        
        #print("AAAA")
        # Define an initial bounding box
        bbox = (287, 183, 46, 42)

        # Uncomment the line below to select a different bounding box
        #bbox = cv2.selectROI(frame, False)

        # Initialize tracker with first frame and bounding box
        ok = self.tracker.init(frame, bbox)

        center_pos = Pose()
        center_pos.pose.orientation = Quaternion(0.,0.,0.,1.)
        center_pos.pose.position.x = 0.0
        center_pos.pose.position.y = 0.0
        center_pos.pose.position.z = 0.0
        while not rospy.is_shutdown():
            
            # Read a new frame
            ok, frame = self.video.read()
            if not ok:
                continue
            
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            #kernal = np.ones((5, 5), "uint8")
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([60,110,100])
            upper_blue = np.array([130,255,255])
        
            # Here we are defining range of bluecolor in HSV
            # This creates a mask of blue coloured
            # objects found in the frame.
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            #cv2.imshow('mask',mask)
            res = cv2.bitwise_and(frame,frame, mask= mask)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            #res = cv2.bitwise_and(frame,frame, mask= mask)
            #gray = res
            #cv2.imshow('frame',frame)
            #cv2.imshow('mask',mask)
            #cv2.imshow('res',res)

            frame = res
            # Start timer
            timer = cv2.getTickCount()
    
            # Update tracker
            ok, bbox = self.tracker.update(frame)
    
            # Calculate Frames per second (FPS)
            #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    
            # Publish center
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

                # Rectangle Center:
                x_center = (p1[0]+p2[0])/2
                y_center = (p1[1]+p2[1])/2
                center_pos.pose.position.x = x_center
                center_pos.pose.position.y = y_center
            self.poi_pub.publish(center_pos)
            #else :
            #    pass
                # Tracking failure
                #cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display tracker type on frame
            #cv2.putText(frame, self.tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
            # Display FPS on frame
            #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
            # Display result
            #cv2.imshow("Tracking", frame)
    
            # Exit if ESC pressed
            #if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            #    break
        #video.release()
        #cv2.destroyAllWindows()

if __name__ == '__main__' :
    rospy.init_node("aabl_track_color_sphero", anonymous=False)
    rospy.loginfo("Starting the aabl track_color_sphero node")
    lookat = TrackSphero()
    rospy.spin()