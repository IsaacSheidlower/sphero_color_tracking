import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture(-1) # for using CAM
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()

# Start timer
timer = cv2.getTickCount()  
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    image = frame
    output = image.copy()
    height, width = image.shape[:2]
    maxRadius = 35#int(10.0*(width/12)/2)
    minRadius = 15#int(0.1*(width/12)/2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(image=gray, 
                            method=cv2.HOUGH_GRADIENT, 
                            dp=1.2, 
                            minDist=2*minRadius,
                            param1=50,
                            param2=50,
                            minRadius=minRadius,
                            maxRadius=maxRadius                           
                            )

          
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # Start timer
    timer = cv2.getTickCount()

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
    
        circlesRound = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        print(circles)
        print("Circle radiuses:")
        for (x, y, r) in circlesRound:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            print("   r", r)

        cv2.imshow("Circles", output)
    else:
        print ('No circles found')
        cv2.imshow("Circles", frame)
        # Exit if ESC pressed

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        break
video.release()
cv2.destroyAllWindows()