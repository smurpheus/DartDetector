#!/usr/bin/python
import sys, getopt
from sys import stdout
import cv2
import math
import numpy as np
from numpy import ones, vstack
from utils import Camera, Board, projectReverse, ContourStorage
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
            print("click in a %s field please" % self.colors.keys()[i])

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
                    cv2.resizeWindow('closed', 640, 480)
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
                    print("Color %s val %s" % (co, val))
                    self.colors[co] = cv2.cvtColor(np.array([[self.f1[y, x]]], np.uint8), cv2.COLOR_BGR2HSV)
                    break
                    # cv2.circle(f1,(x,y),100,(255,0,0),-1)

    def _fixbdiff(self, val):
        self.bdiff = val

    def _fixgdiff(self, val):
        self.gdiff = val

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
                    upper = np.array(
                        [self.color[0] + self.bdiff, self.color[1] + self.gdiff, self.color[2] + self.rdiff])
                    lower = np.array(
                        [self.color[0] - self.bdiff, self.color[1] - self.gdiff, self.color[2] - self.rdiff])
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
    ###Default ######
    history = 500
    shad_tresh = 0.5
    var_tresh = 16
    var_max = 75
    var_min = 4
    #################
    history = 500
    shad_tresh = 0.35
    var_tresh = 15
    var_max = 75
    var_min = 1
    fgbg = None
    def _initialize_substractor(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.fgbg.setHistory(self.history)
        self.fgbg.setShadowThreshold(self.shad_tresh)
        self.fgbg.setVarThreshold(self.var_tresh)
        self.fgbg.setVarMax(self.var_max)
        self.fgbg.setVarMin(self.var_min)
        return self.fgbg

    def __init__(self, c1):
        print("BackgroundSubstractor called with capture %s" % c1)
        self.camera = Camera(device=c1)
        # c1.release()re('test.avi')
        background,reseted = self.camera.get_image()
        # cv2.namedWindow("Background", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Current", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("FG Substraction", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("History", "Current", self.history, 1000, self._set_history)
        cv2.createTrackbar("Shadow Treshold", "Current", int(self.shad_tresh*100), 100, self._set_shad_tresh)
        cv2.createTrackbar("VarThreshold", "Current", self.var_tresh, 100, self._set_var_tresh)
        cv2.createTrackbar("VarMax", "Current", self.var_max, 100, self._set_var_max)
        cv2.createTrackbar("VarMin", "Current", self.var_min, 100, self._set_var_min)
        # cv2.namedWindow("Simple Diff", cv2.WINDOW_NORMAL)
        # cv2.imshow("Background", background)

        # self.fgbg = cv2.createBackgroundSubtractorKNN(history=self.history)
        storage = ContourStorage()


        from matplotlib import pyplot as plt
        # template = cv2.imread('../resources/arrowtemplate.jpg', 0)
        # orb = cv2.ORB_create()
        #
        # # find the keypoints and descriptors with SIFT
        # kp1, des1 = orb.detectAndCompute(template, None)
        # # print kp1
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # for each in kp1:
        #     print each.angle, each.pt, each.octave
        #     cv2.circle(template, (int(each.pt[0]), int(each.pt[1])),5,(0,0,255), 1)
        # # print des1
        # cv2.imshow("Original", template)
        # cv2.waitKey(-1)


        self._initialize_substractor()
        while (True):
            f1,reseted = self.camera.get_image()
            if reseted:
                self._initialize_substractor()


            # kp2, des2 = orb.detectAndCompute(f1, None)
            # # Match descriptors.
            # matches = bf.match(des1, des2)
            #
            # # Sort them in the order of their distance.
            # matches = sorted(matches, key=lambda x: x.distance)
            #
            # # Draw first 10 matches.
            # img3 = cv2.drawMatches(template, kp1, f1, kp2, matches[:100], flags=2, outImg=f1)
            # cv2.imshow("Original", img3)
            # cv2.waitKey(-1)


            # diff = cv2.absdiff(background, f1)

            fgmask1 = self.fgbg.apply(f1)
            fgmask1 = cv2.inRange(fgmask1, 250, 255)
            kernel = np.ones((3,3), np.uint8)
            opened = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((20, 20), np.uint8)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            closed2 = np.array(closed)
            im2, contours, hierarchy = cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(f1, contours, -1,(0,255,0),-1)
            colored = cv2.cvtColor(closed2, cv2.COLOR_GRAY2BGR)
            # contours = [c for c in contours if cv2.contourArea(c) > 1000]
            storage.add_to_storage(contours, f1)
            # img, contours, maxc = storage.get_biggest_contour_image()
            contours = storage.get_best_contours(history=self.history)
            stdout.flush()
            for cnt in contours:
                if len(cnt) > 100:
                    cv2.drawContours(colored, [cnt], 0, (0, 255, 0), -1)
                    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                    cv2.drawContours(colored, [approx], 0, (255, 255, 0), 2)

                    stdout.flush()
                    # print "ratio: %s" % aspect_ratio
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    b = box
                    cv2.circle(colored, (b[0][0], b[0][1]), 4, [0,255,255], 2)
                    cv2.circle(colored, (b[1][0], b[1][1]), 4, [0,255,255], 2)
                    cv2.circle(colored, (b[2][0], b[2][1]), 4, [255,0,255], 2)
                    dist1 = np.linalg.norm(b[0] - b[1])
                    dist2 = np.linalg.norm(b[1] - b[2])
                    # cv2.circle(colored, (b[3][0], b[3][1]), 4, [0,255,255], 2)
                    box = np.int0(box)
                    cv2.drawContours(colored, [box], 0, (0, 0, 255), 2)
                    rows, cols = colored.shape[:2]
                    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)
                    # print (cols - 1, righty), (0, lefty)
                    cimg = np.zeros_like(colored)
                    cv2.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
                    # Access the image pixels and create a 1D numpy array then add to list
                    pts = np.where(cimg == 255)
                    pts = zip(pts[1], pts[0])
                    pts_on_line = []
                    for pt in pts:
                        x, y = pt
                        ly = int(np.interp(int(x), [0, cols - 1], [lefty, righty]))
                        if y < ly +1 and y > ly -1:
                            pts_on_line.append(pt)
                    cv2.line(colored, (cols - 1, righty), (0, lefty), (255, 255, 0), 1)
                    # ellipse = cv2.fitEllipse(cnt)
                    # cv2.ellipse(colored, ellipse, (0, 255, 0), 2)
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(colored,(cx,cy), 5, [255, 0, 0], 2)
                    colored[cy, cx] = [0, 0, 255]
                    hull = cv2.convexHull(cnt)

                    centroid = np.array([cx,cy])
                    mdist = None
                    mpt = None
                    for pt in pts_on_line:
                        p = np.array(pt)
                        dist = np.linalg.norm(centroid-p)
                        if mdist is None or dist > mdist:
                            mdist = dist
                            mpt = p
                    stdout.write("\rratio: %s Box points: %s  Maxpoint: %s  Arclen: %s" % ((dist1/dist2), cv2.contourArea(box), mpt, cv2.arcLength(cnt, True)))

                    if not mpt is None:
                        cv2.circle(colored, (mpt[0], mpt[1]), 3, [255, 0, 0], 2)
                        colored[mpt[1], mpt[0]] = [0, 0, 255]
                        cv2.circle(f1, (mpt[0], mpt[1]), 3, [255, 0, 0], 2)
                        f1[mpt[1], mpt[0]] = [0, 0, 255]

                # area = cv2.contourArea(cnt)
            cv2.imshow("Current", closed)
            cv2.imshow("FG Substraction", colored)
            cv2.imshow("Original", f1)

            # fgmask = self.fgbg.apply(f1)
            # ret, thresh = cv2.threshold(fgmask, 127, 255, 0)
            #
            # contours,_,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # # cv2.drawContours(f1, contours, -1, (0, 255, 0), 3)
            # closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            # cv2.imshow("Current", contours)

            # cv2.imshow("Simple Diff", diff)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == 119:
                cv2.waitKey(-1)
        cv2.destroyAllWindows()
    def _set_history(self, val):
        self.history = val
        self.fgbg.setHistory(self.history)

    def _set_shad_tresh(self, val):
        self.shad_tresh = val/100.
        self.fgbg.setShadowThreshold(self.shad_tresh)


    def _set_var_tresh(self, val):
        self.var_tresh = val
        self.fgbg.setVarThreshold(self.var_tresh)

    def _set_var_min(self, val):
        self.var_min = val
        self.fgbg.setVarMin(self.var_min)

    def _set_var_max(self, val):
        self.var_max = val
        self.fgbg.setVarMax(self.var_max)


class BoardCalibrator(object):
    imgpoints = []

    def __init__(self, input):
        camera = Camera(device=input)
        camera.do_calibration(img=True)
        self.frame,reseted  = camera.get_image()
        # self.frame = camera.undistort_image(self.frame)
        cv2.namedWindow("Calibration Window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Calibration Hint", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Window", self.frame)
        cv2.waitKey(1)

        allobj = Board()._get_configs()
        [i.append(0) for i in allobj]
        nobj = np.array(allobj, np.float64)
        config_points = nobj
        cv2.setMouseCallback("Calibration Window", self.click)
        pos = 0
        for i in config_points:
            print ("Select field %s please. Accept with any key." % str(i))
            hintim = Board().get_config_hint(pos)
            cv2.imshow("Calibration Hint", hintim)
            k = cv2.waitKey(-1) & 0xFF
            self.imgpoints.append(self.imgpoint)
            if k == 27:
                print("Escaped and closing.")
                break
            else:
                print("Thank you")
                pos += 1

        print("Imagepoints %s" % self.imgpoints)

        print("Objp %s" % nobj)
        _, rvec, tvec = cv2.solvePnP(nobj,
                                     np.array(self.imgpoints, np.float64), np.array(camera.config['mtx']),
                                     np.array(camera.config['dist']), None,None, False, cv2.SOLVEPNP_ITERATIVE)
        mean_error = 0
        imgpoints2, _ = cv2.projectPoints(nobj, rvec, tvec, camera.cameramatrix, camera.config['dist'])
        impoints = np.array([[x] for x in self.imgpoints], np.float64)
        error = cv2.norm(impoints, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

        print "total error: ", mean_error
        outers = Board().get_radius()
        def calcy(x,r):
            return np.sqrt(np.power(r,2)-np.power(x,2))

        points = []
        for outer in outers:
            xs = [i*0.1 for i in range(0,int(outer*10))]
            xs += [i*-0.1 for i in range(0,int(outer*10))]
            for i in xs:
                y = calcy(i,int(outer))
                points.append([[i],[y],[0]])
                points.append([[i], [-y], [0]])
        points = np.array(points)
        # points = np.array([[[0],[0],[0]], [[outer],[0],[0]]])
        print points
        imp, jac = cv2.projectPoints(points, rvec,tvec, np.array(camera.config['mtx']), np.array(camera.config['dist']))
        print imp
        print("NEWRVEC: %s" % rvec)
        print("NEWTVEC: %s" % tvec)
        rot, _ = cv2.Rodrigues(rvec)
        rev = projectReverse(self.imgpoints,rot, tvec, camera.config['mtx'])
        self.rot = rot
        self.tvec = tvec
        self.camera = camera
        # b.draw_board_to_frame(frame2)
        print("Imagepoints %s" % self.imgpoints)
        print("Objecpoints %s" % rev)

        cv2.setMouseCallback("Calibration Window", self.calcObj)
        while True:
            self.frame,reseted  = camera.get_image()
            self.frame = camera.undistort_image(self.frame)
            for i in imp:
                try:
                    self.frame[i[0][1], i[0][0]] = [0,0,255]
                except IndexError:
                    pass
            # self.frame = cv2.circle(self.frame,(int(np.round(imp[0][0][0])), int(np.round(imp[0][0][1]))),int(np.round(dist)),[0,0,255],1)
            cv2.imshow("Calibration Window", self.frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        k = cv2.waitKey(-1) & 0xFF

    def calcObj(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked %s: %s" % (x, y))

            objp = projectReverse(np.array([[[x],[y]]]), self.rot, self.tvec, self.camera.config['mtx'])
            print "Object Point %s"%objp
            print "Field: %s" %Board().calculate_field(objp[0][:2])


    def click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked %s: %s" % (x, y))
            self.imgpoint = [x,y]
            # self.imgpoints.append([x, y])

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

            # c1 = cv2.VideoCapture(inputfile)
            # # width = c1.get(3)
            # # height = c1.get(4)
            #
            # able_to_read, f1 = c1.read()
            # hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
            # print(able_to_read)
            # cc = CountourDetector(c1)
            bs = BackgroundSubtractor(inputfile)
            # bd = BlobDetector(c1)
            # bc = BoardCalibrator(inputfile)
        elif opt in ("-d", "--device"):
            if arg == "":
                device = 0
            device = int(arg)

            print "With device called"
            # c1 = cv2.VideoCapture(1)
            # c1.set(3, width)
            # c1.set(4, height)
            # able_to_read, f1 = c1.read()
            # hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
            # print(able_to_read)
            # cc = CountourDetector(c1)
            bs = BackgroundSubtractor(device)
            # bd = BlobDetector(c1)
            # bc = BoardCalibrator(device)
    print ('Output file is "', inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
