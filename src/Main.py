import cv2
import numpy as np

c1 = cv2.VideoCapture('../resources/output1.avi')
color = None


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print "Called %s %s" % (x, y)
        print f1[y, x]
        global color
        color = hsv[y, x]
        # cv2.circle(f1,(x,y),100,(255,0,0),-1)


while (True):
    able_to_read, f1 = c1.read()
    if able_to_read:
        hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
        cv2.imshow("Original", f1)
        asd = cv2.setMouseCallback('Original', draw_circle)
        if color is not None:
            upper = np.array([color[0] + 20, color[1] + 50, color[2] + 50])
            lower = np.array([color[0] - 20, color[1] - 50, color[2] - 50])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((8, 8), np.uint8)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            cv2.imshow('mask', mask)
            cv2.imshow('closed', closed)
            cv2.imshow('opened', opened)


        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        c1 = cv2.VideoCapture('../resources/output1.avi')
cv2.destroyAllWindows()
