import cv2
import numpy as np

c1 = cv2.VideoCapture(0)
color = None
able_to_read, f1 = c1.read()
hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)

# def draw_circle(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print "Called %s %s" % (x, y)
#         print f1[y, x]
#         global color
#         color = hsv[y, x]
#         # cv2.circle(f1,(x,y),100,(255,0,0),-1)

# while (True):
#     able_to_read, f1 = c1.read()
#     if able_to_read:
#         hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
#         cv2.imshow("Original", f1)
#         cv2.setMouseCallback('Original', draw_circle)
#         params = cv2.SimpleBlobDetector_Params()
#         params.filterByArea = True
#         params.minArea = 20
#         params.filterByInertia = True
#         params.minInertiaRatio = 0.001
#         params.filterByColor = True
#         params.blobColor = 255
#         # params.minRepeatability = 20
#         # params.minDistBetweenBlobs = 8
#         params.filterByConvexity = True
#         params.minConvexity = 0.6
#
#         detector = cv2.SimpleBlobDetector_create(params)
#
#         if color is not None:
#             upper = np.array([color[0] + 20, color[1] + 60, color[2] + 60])
#             lower = np.array([color[0] - 20, color[1] - 60, color[2] - 60])
#             mask = cv2.inRange(hsv, lower, upper)
#             kernel = np.ones((8, 8), np.uint8)
#             closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#             opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
#             keypoints = detector.detect(closed)
#             im_with_keypoints = cv2.drawKeypoints(closed, keypoints, np.array([]), (0, 0, 255),
#                                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#             #cv2.imshow('mask', mask)
#             cv2.imshow('closed', opened)
#             cv2.imshow('Keypoints', im_with_keypoints)
#
#
#         k = cv2.waitKey(5) & 0xFF
#         if k == 27:
#             break
#     else:
#         c1 = cv2.VideoCapture('../resources/output1.avi')
# cv2.destroyAllWindows()

class BoardCalibrator(object):
    def __init__(self, frame):
        self.field_angle = {1: 18, 2: 144, 3: 180, 4: 54, 5: 342, 6: 90, 7: 216, 8: 252, 9: 306, 10: 108, 11: 270,
                            12: 324, 13: 72, 14: 288, 15: 126, 16: 234, 17: 162, 18: 36, 19: 198, 20: 0}
        self.frame = frame
        self.clicked = False
        self.fields = {20: None, 3: None, 6: None, 11: None, 'mid': None}
        cv2.imshow("Calibration Window", frame)
        for i in range(len(self.fields.keys())):
            cv2.setMouseCallback("Calibration Window", self._clickedIntoPicture, i)
            print ("Select field %s please. Accept with any key." % (self.fields.keys()[i]))
            k = cv2.waitKey(-1) & 0xFF
            if k == 27:
                print("Escaped and closing.")
                break
            else:
                print "Thank you"
        print self.fields

        from numpy import ones, vstack
        from numpy.linalg import lstsq
        points = [(self.fields[20]['x'], self.fields[20]['y']), (self.fields['mid']['x'], self.fields['mid']['y'])]
        x_coords, y_coords = zip(*points)
        print x_coords
        print y_coords
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        x1 = (0 - c) / m
        print "x: %s , y: %s" % (x1, 0)
        x2 = (480 - c) / m
        print "x: %s , y: %s" % (x2, 480)
        cv2.line(self.frame, (int(x1), 0), (int(x2), 480), (255,0,0), 5)
        cv2.imshow("Calibration Window", frame)
        k = cv2.waitKey(-1) & 0xFF
        print "Line Solution is y = {m}x + {c}".format(m=m, c=c)


    def _clickedIntoPicture(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            field_key = self.fields.keys()[param]
            print self.frame[y, x]
            self.fields[field_key] = {'y': y, 'x': x}



bc = BoardCalibrator(f1)
