# !/usr/bin/python
import sys, getopt
import cv2
import math
import numpy as np
from numpy import ones, vstack
from utils import Camera, Board
import time
from numpy.linalg import lstsq

width = 1280
height = 960
# width = 640
# height = 480
history = 500
open_close_mask = 5


class CountourDetector(object):
    def _get_color(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Called %s %s" % (x, y))
            print(self.f1[y, x])
            self.color = self.hsv[y, x]
            # cv2.circle(f1,(x,y),100,(255,0,0),-1)

    def __init__(self, c1):
        able_to_read, f1 = c1.read()
        self.f1 = f1
        self.hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
        self.color = None
        self.colors = {'red': None, 'green': None, 'black': None, 'white': None}
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("closed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
        cv2.imshow("Original", f1)
        cv2.setMouseCallback('Original', self._get_color)
        for i in range(4):
            print("click in a %s field please" %self.colors.keys()[i])

        while (True):
            able_to_read, f1 = c1.read()
            if able_to_read:
                cv2.imshow("Original", f1)
                cv2.resizeWindow("Original", 640, 480)
                self.f1 = f1
                self.hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = True
                params.minArea = 20
                params.filterByInertia = True
                params.minInertiaRatio = 0.001
                params.filterByColor = True
                params.blobColor = 255
                # params.minRepeatability = 20
                # params.minDistBetweenBlobs = 8
                params.filterByConvexity = True
                params.minConvexity = 0.6

                detector = cv2.SimpleBlobDetector_create(params)

                if self.color is not None:
                    upper = np.array([self.color[0] + 20, self.color[1] + 60, self.color[2] + 60])
                    lower = np.array([self.color[0] - 20, self.color[1] - 60, self.color[2] - 60])
                    mask = cv2.inRange(self.hsv, lower, upper)
                    kernel = np.ones((open_close_mask, open_close_mask), np.uint8)
                    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                    canny = cv2.Canny(opened, 0, 255)
                    im2, contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # for cnt in contours[1:2]:
                    #     print ("Countur %s "%contours.index(cnt))
                    # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                    # print len(approx)
                    # if len(approx) == 5:
                    #     print "pentagon"
                    #     cv2.drawContours(canny, [cnt], 0, 255, -1)
                    # elif len(approx) == 3:
                    #     print "triangle"
                    #     cv2.drawContours(canny, [cnt], 0, (0, 255, 0), -1)
                    # elif len(approx) == 4:
                    #     print "square"
                    #     cv2.drawContours(canny, [cnt], 0, (0, 0, 255), -1)
                    # elif len(approx) == 9:
                    #     print "half-circle"
                    #     cv2.drawContours(canny, [cnt], 0, (255, 255, 0), -1)
                    # elif len(approx) > 15:
                    #     print "circle"
                    asd = cv2.drawContours(f1, contours, -1, (0, 0, 255), -1)
                    keypoints = detector.detect(closed)
                    im_with_keypoints = cv2.drawKeypoints(closed, keypoints, np.array([]), (0, 0, 255),
                                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # cv2.imshow('mask', mask)
                    cv2.imshow('closed', asd)
                    cv2.resizeWindow('closed', 640,480)
                    cv2.imshow('Keypoints', im_with_keypoints)
                    cv2.resizeWindow('Keypoints', 640, 480)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                c1.set(1, 0)
        cv2.destroyAllWindows()


class BlobDetector(object):
    def _get_color(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Called %s %s" % (x, y))
            print(self.f1[y, x])
            self.color = self.hsv[y, x]
            for co, val in self.colors.iteritems():
                if val is None:
                    print("Color %s val %s" %(co,val))
                    self.colors[co] = cv2.cvtColor(np.array([[self.f1[y, x]]],np.uint8), cv2.COLOR_BGR2HSV)
                    break
            # cv2.circle(f1,(x,y),100,(255,0,0),-1)
    def _fixbdiff(self, val):
        self.bdiff = val

    def _fixgdiff(self, val):
        self.gdiff =val

    def _fixrdiff(self, val):
        self.rdiff = val

    def __init__(self, c1):
        self.color = None
        able_to_read, f1 = c1.read()
        self.f1 = f1
        self.hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
        self.colors = {'red': None, 'green': None, 'black': None, 'white': None}
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("closed", cv2.WINDOW_NORMAL)
        self.bdiff = 40
        self.gdiff = 40
        self.rdiff = 40
        cv2.createTrackbar("bdiff", "closed", 120, 255, self._fixbdiff)
        cv2.createTrackbar("gdiff", "closed", 120, 255, self._fixgdiff)
        cv2.createTrackbar("rdif", "closed", 120, 255, self._fixrdiff)
        cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
        cv2.imshow("Original", f1)
        cv2.setMouseCallback('Original', self._get_color)
        for i in range(4):
            print("click in a %s field please" % self.colors.keys()[i])
            k = cv2.waitKey(-1) & 0xFF
            if k == 27:
                break
        print(self.colors)
        while (True):
            able_to_read, f1 = c1.read()
            if able_to_read:
                self.f1 = f1
                cv2.imshow("Original", f1)
                self.hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)

                # cv2.setMouseCallback('Original', draw_circle)
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = True
                params.minArea = 20
                params.filterByInertia = True
                params.minInertiaRatio = 0.001
                params.filterByColor = True
                params.blobColor = 255
                # params.minRepeatability = 20
                # params.minDistBetweenBlobs = 8
                params.filterByConvexity = True
                params.minConvexity = 0.6

                detector = cv2.SimpleBlobDetector_create(params)

                if self.color is not None:
                    upper = np.array([self.color[0] + self.bdiff, self.color[1] + self.gdiff, self.color[2] + self.rdiff])
                    lower = np.array([self.color[0] - self.bdiff, self.color[1] - self.gdiff, self.color[2] - self.rdiff])
                    mask = cv2.inRange(self.hsv, lower, upper)
                    kernel = np.ones((open_close_mask, open_close_mask), np.uint8)
                    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                    corners = cv2.cornerHarris(opened, 9, 9, 5)
                    canny = cv2.Canny(opened, 0, 255)
                    keypoints = detector.detect(closed)
                    im_with_keypoints = cv2.drawKeypoints(closed, keypoints, np.array([]), (0, 0, 255),
                                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # cv2.imshow('mask', mask)
                    cv2.imshow('closed', corners)
                    cv2.imshow('Keypoints', im_with_keypoints)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
            else:
                c1.set(1, 0)
        cv2.destroyAllWindows()


class BackgroundSubtractor(object):
    def __init__(self, c1):
        print("BackgroundSubstractor called")
        self.camera = Camera()
        # c1.release()re('test.avi')
        able_to_read, background = c1.read()
        cv2.namedWindow("Background", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Current", cv2.WINDOW_NORMAL)
        cv2.namedWindow("FG Substraction", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Simple Diff", cv2.WINDOW_NORMAL)
        cv2.imshow("Background", background)

        fgbg = cv2.createBackgroundSubtractorMOG2(history=history)
        while (True):
            able_to_read, f1 = c1.read()
            if able_to_read:
                diff = cv2.absdiff(background, f1)
                fgmask = fgbg.apply(f1)
                kernel = np.ones((open_close_mask, open_close_mask), np.uint8)
                closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                cv2.imshow("Current", f1)
                cv2.imshow("FG Substraction", opened)
                cv2.imshow("Simple Diff", diff)
                time.sleep(0.05)
            else:
                c1.set(1, 0)
                fgbg = cv2.createBackgroundSubtractorMOG2(history=history)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


class BoardCalibrator(object):
    imgpoints = []
    def __init__(self, inputframe):
        self.field_angle = {1: 18, 2: 144, 3: 180, 4: 54, 5: 342, 6: 90, 7: 216, 8: 252, 9: 306, 10: 108, 11: 270,
                            12: 324, 13: 72, 14: 288, 15: 126, 16: 234, 17: 162, 18: 36, 19: 198, 20: 0}
        self.frame = inputframe
        from utils import Camera
        camera = Camera(width=680,heigth=480)
        camera.load_config("new_conf.json")
        self.frame = camera.undistort_image(self.frame)
        frame2 = camera.undistort_image_without_crop(self.frame)
        self.clicked = False
        self.fields = {20: None, 3: None, 6: None, 11: None, 'mid': None}
        # cv2.namedWindow("Uncropped", cv2.WINDOW_NORMAL)
        radius = int(camera.height/2) - 100
        self.after = {'mid': {'y': int(camera.height/2), 'x': int(camera.width/2)},
                      20: {'y': 100, 'x': int(camera.width/2)},
                      3: {'y': int(camera.height - 100), 'x': int(camera.width/2)},
                      11: {'x': int(camera.width/2) - radius, 'y': int(camera.height/2)},
                      6: {'x': int(camera.width/2) + radius, 'y': int(camera.height/2)}}
        self.after.pop(11)
        self.after.pop(3)
        cv2.namedWindow("Calibration Window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("CONTROL Window", cv2.WINDOW_NORMAL)
        cv2.imshow("CONTROL Window", frame2)
        cv2.imshow("Calibration Window", self.frame)
        cv2.waitKey(1)
        for i in range(len(self.fields.keys())):
            cv2.setMouseCallback("Calibration Window", self._clickedIntoPicture, i)
            print ("Select field %s please. Accept with any key." % (self.fields.keys()[i]))
            k = cv2.waitKey(-1) & 0xFF
            if k == 27:
                print("Escaped and closing.")
                break
            else:
                print("Thank you")



        print(self.fields)
        before = []
        after = []

        for k, v in self.after.iteritems():
            after.append([v['x'], v['y']])
            before.append([self.fields[k]['x'],self.fields[k]['y']])
        print  "AWESOME POINTS MATE"
        print before
        print after
        M = cv2.getAffineTransform(np.float32(before[:3]), np.float32(after[:3]))
        print "M: %s"%M
        rows, cols, _ = self.frame.shape
        frame2 = cv2.warpAffine(self.frame, M, (cols,rows))
        b = Board(radius, (self.after['mid']['x'], self.after['mid']['y']))

        for i in b.get_corners():
            cv2.setMouseCallback("Calibration Window", self.click)
            print ("Select field %s please. Accept with any key." %str(i))
            k = cv2.waitKey(-1) & 0xFF
            if k == 27:
                print("Escaped and closing.")
                break
            else:
                print("Thank you")
        b.draw_board_to_frame(frame2)
        print("Imagepoints %s"%self.imgpoints)
        for xm,ym in after:
            cv2.circle(frame2, (xm, ym), int(5), [0, 255, 255])
        cv2.imshow("CONTROL Window", frame2)
        points = [(self.fields[20]['x'], self.fields[20]['y']), (self.fields['mid']['x'], self.fields['mid']['y'])]
        xdiff = abs(points[0][0] - points[1][0])
        ydiff = abs(points[0][1] - points[1][1])
        print  xdiff
        print ydiff
        print xdiff**2 + ydiff**2
        c = (xdiff**2 + ydiff**2)**(1./2.)
        print c
        # for i in board:
        #     cv2.circle(self.frame, (self.fields['mid']['x'], self.fields['mid']['y']), int(c * i), [0, 0, 255])

        x1, x2 = self._get_line_in_pic(points)
        cv2.line(self.frame, (int(x1), 0), (int(x2), int(height)), (255, 0, 0), 2)
        for field, fangle in self.field_angle.iteritems():
            rx, ry = self._rotate_point(np.array([x1 - self.fields['mid']['x'], 0 - self.fields['mid']['y']]), fangle)
            rx = rx + self.fields['mid']['x']
            ry = ry + self.fields['mid']['y']
            trans = self._make_rotation_transformation(math.radians(fangle), (self.fields['mid']['x'], self.fields['mid']['y']))
            rx, ry = trans((x1, 0))
            print("rx %s ry %s" % (rx, ry))
            points = [(rx, ry), (self.fields['mid']['x'], self.fields['mid']['y'])]
            rx1, rx2 = self._get_line_in_pic(points)

            cv2.line(self.frame, (int(rx1), 0), (int(rx2), int(height)), (255, 0, 0), 1)
        cv2.imshow("Calibration Window", self.frame)
        k = cv2.waitKey(-1) & 0xFF
        # print "Line Solution is y = {m}x + {c}".format(m=m, c=c)

    def click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked %s: %s" % (x, y))
            self.imgpoints.append([x, y])

    def _get_line_in_pic(self, points):
        x_coords, y_coords = zip(*points)
        # print (x_coords)
        # print (y_coords)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        x1 = (0 - c) / m
        # print ("x: %s , y: %s" % (x1, 0))
        x2 = (height - c) / m
        # print("x: %s , y: %s" % (x2, height))
        return x1, x2

    def _make_rotation_transformation(self, angle, origin=(0, 0)):
        cos_theta, sin_theta = math.cos(angle), math.sin(angle)
        x0, y0 = origin

        def xform(point):
            x, y = point[0] - x0, point[1] - y0
            return (x * cos_theta - y * sin_theta + x0,
                    x * sin_theta + y * cos_theta + y0)

        return xform


    def _rotate_point(self, point, rangle):
        b = point[1]  # hieght
        # print("a: %s" % b)
        a = point[0]  #
        # print("b %s" % a)
        angle = math.atan(a / b)
        # print("angle %s" % math.degrees(angle))
        dangle = math.degrees(angle)
        c = b / math.cos(angle)
        # print ("c %s" % c)
        ndangle = dangle + rangle
        nangle = math.radians(ndangle)
        na = math.sin(nangle) * c
        # print("New a %s" % na)
        nb = math.cos(nangle) * c
        # print("New b %s" % nb)
        return na, nb

    def _clickedIntoPicture(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            field_key = self.fields.keys()[param]
            print(self.frame[y, x])
            self.fields[field_key] = {'y': y, 'x': x}


def main(argv):
    inputfile = ''
    width = 1280
    height = 960
    try:
        opts, args = getopt.getopt(argv, "h:i:d", ["ifile=", "device="])
    except getopt.GetoptError:
        print ('utils.py -i <inputputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('utils.py -i <inputputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

            c1 = cv2.VideoCapture(inputfile)
            width = c1.get(3)
            height = c1.get(4)

            able_to_read, f1 = c1.read()
            hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
            print(able_to_read)
            cc = CountourDetector(c1)
            bs = BackgroundSubtractor(c1)
            bd = BlobDetector(c1)
            bc = BoardCalibrator(f1)
            c1.release()
        elif opt in ("-d", "--device"):
            c1 = cv2.VideoCapture(1)
            c1.set(3, width)
            c1.set(4, height)
            able_to_read, f1 = c1.read()
            hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
            print(able_to_read)
            cc = CountourDetector(c1)
            bs = BackgroundSubtractor(c1)
            bd = BlobDetector(c1)
            bc = BoardCalibrator(f1)
    print ('Output file is "', inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])



