# !/usr/bin/python
import sys, getopt
from sys import stdout
import cv2
import math
import numpy as np
from numpy import ones, vstack
from utils import Camera, Board, projectReverse, ContourStorage
import time
from numpy.linalg import lstsq
from pygame import mixer
from threading import Thread, Lock
import copy
import csv
import os.path
import json
from matplotlib import pyplot as plt
from datetime import datetime

width = 1280
height = 960
# width = 640Lock
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


class NoImage(Exception):
    pass


class BackgroundSubtractor(Thread):
    ###Default ######
    history = 500
    shad_tresh = 0.5
    var_tresh = 16
    var_max = 75
    var_min = 4
    #################
    history = 200
    shad_tresh = 0.55
    var_tresh = 16
    var_max = 75
    var_min = 1
    arrows = []
    fgbg = None
    threadLock = None
    storage = None
    image = None
    stopped = False
    paused = False

    def _initialize_substractor(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.fgbg.setHistory(self.history)
        self.fgbg.setShadowThreshold(self.shad_tresh)
        self.fgbg.setVarThreshold(self.var_tresh)
        # self.fgbg.setVarMax(self.var_max)
        # self.fgbg.setVarMin(self.var_min)
        return self.fgbg

    def get_image(self):
        self.threadLock.acquire()
        img = copy.copy(self.image)
        self.threadLock.release()
        return img

    def set_image(self, img):
        self.threadLock.acquire()
        self.image = img
        self.threadLock.release()

    def get_substracted(self):
        self.threadLock.acquire()
        img = copy.copy(self.substracted)
        self.threadLock.release()
        return img

    def set_substracted(self, img):
        self.threadLock.acquire()
        self.substracted = img
        self.threadLock.release()

    def _set_arrow(self, arrow):
        self.threadLock.acquire()
        self.arrows.append(arrow)
        self.threadLock.release()

    def clear_arrows(self):
        self.threadLock.acquire()
        self.arrows = []
        self.threadLock.release()

    def get_arrows(self):
        self.threadLock.acquire()
        return_list = list(self.arrows)
        self.threadLock.release()
        return return_list

    def get_storage(self):
        self.threadLock.acquire()
        return_list = list(self.storage.storage)
        unaltered = list(self.storage.unaltered)
        self.threadLock.release()
        return return_list, unaltered

    def _add_to_storage(self, contours, f1, no_of_frame):
        self.threadLock.acquire()
        self.storage.add_to_storage(contours, f1, no_of_frame)
        self.threadLock.release()

    def __init__(self, c1=0, camera=None):
        Thread.__init__(self)
        self.threadLock = Lock()
        print("BackgroundSubstractor called with capture %s" % c1)
        if not isinstance(camera, Camera):
            self.camera = Camera(device=c1)
        else:
            self.camera = camera
        # c1.release()re('test.avi')

        self.storage = ContourStorage()
        self._initialize_substractor()

    def run(self):
        try:
            self.run_substraction()
        except NoImage:
            self.stopped = True

    def run_substraction(self):

        while not self.stopped:
            if not self.paused:
                res = self.camera.get_image()
                if res is not None:
                    f1, reseted = res
                else:
                    raise NoImage()
                no_of_frame = self.camera.read_frame_no
                if reseted:
                    self._initialize_substractor()

                fgmask1 = self.fgbg.apply(f1)
                fgmask1 = cv2.inRange(fgmask1, 250, 255)
                kernel = np.ones((6, 6), np.uint8)
                closed = cv2.morphologyEx(fgmask1, cv2.MORPH_CLOSE, kernel)
                kernel = np.ones((4, 4), np.uint8)
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                closed2 = np.array(closed)
                self.set_substracted(closed)
                contours, hierarchy = cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                colored = cv2.cvtColor(closed2, cv2.COLOR_GRAY2BGR)
                self._add_to_storage(contours, f1, no_of_frame)
                arrow = self.storage.get_arrow(self.history)
                for a in arrow:
                    self._set_arrow(a)
                # print str("Arrows: %s"%self.get_arrows())
                stdout.flush()
                for arrow in self.get_arrows():
                    cv2.drawContours(f1, arrow.contours, -1, (0, 255, 0), -1)
                    cv2.drawContours(f1, [arrow.aproximated], 0, (255, 255, 0), 2)

                    cv2.circle(f1, (arrow.tip[0], arrow.tip[1]), 3, [255, 0, 0], 2)
                    f1[arrow.tip[1], arrow.tip[0]] = [255, 0, 0]

                    cv2.circle(f1, (arrow.tip2[0], arrow.tip2[1]), 3, [101, 8, 108], 2)
                    f1[arrow.tip2[1], arrow.tip2[0]] = [101, 8, 108]

                    cv2.circle(f1, (arrow.tip3[0], arrow.tip3[1]), 3, [225, 97, 53], 2)
                    f1[arrow.tip2[1], arrow.tip2[0]] = [225, 97, 53]

                    cv2.circle(f1, (arrow.tip4[0], arrow.tip4[1]), 3, [81, 106, 37], 2)
                    f1[arrow.tip2[1], arrow.tip2[0]] = [81, 106, 37]

                    rows, cols = f1.shape[:2]
                    cv2.line(f1, (cols - 1, arrow.line[1]), (0, arrow.line[0]), (255, 255, 0), 1)
                    cv2.drawContours(f1, [np.int0(arrow.bbox)], 0, (0, 0, 255), 2)

                self.set_image(f1)
            else:
                time.sleep(1)
        #
        #     cv2.imshow("Current", closed)
        #     cv2.imshow("FG Substraction", colored)
        #     cv2.imshow("Original", f1)
        #
        #     k = cv2.waitKey(1) & 0xFF
        #     if k == ord('f') or len(arrows) >= 3:
        #         arrows = []
        #     if k == 27:
        #         break
        #     if k == 119:
        #         cv2.waitKey(-1)
        # cv2.destroyAllWindows()

    def _set_history(self, val):
        self.history = val
        self.fgbg.setHistory(self.history)

    def _set_shad_tresh(self, val):
        self.shad_tresh = val / 100.
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


class MainApplikacation(object):
    detected = []
    detected2 = []
    detected3 = []
    detected4 = []
    detected5 = []
    real = []
    was_covert = []
    frame_no = []
    Calibrated = None
    Substractor = None
    input = None
    camera = None
    board_config_load = True
    date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    fname = "data%s" % date
    imgpoint = None
    boardpoint = None

    def write_data(self):
        print
        "Writing data to file"
        i = 0
        fname = self.fname
        fname += ".csv"
        while os.path.isfile(fname):
            i += 1
            fname = self.fname + '-' + str(i) + ".csv"
        with open(fname, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', dialect='excel',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                ['Detected', 'Detected2', 'Detected3', 'Detected4', 'Detected5', 'Reality', 'Was Covert', 'Frameno',
                 'Diff', 'Diff2', 'Diff3', 'Diff4', 'Diff5'])

            def calc_diff(a, b):
                i = 0
                diff = []
                for each in a:
                    try:
                        if each == b[i]:
                            diff.append(True)
                        else:
                            diff.append(False)
                    except:
                        print("For some Reasion there was an error in diff calc")
                    i += 1
                percentage = float(len([x for x in diff if x is True])) / float(len(diff))
                return diff, percentage

            diff, percentage = calc_diff(self.detected, self.real)
            print("Tip Version 1 was %s Percent correct." % (percentage * 100))
            diff2, percentage = calc_diff(self.detected2, self.real)
            print("Tip Version 2 was %s Percent correct." % (percentage * 100))
            diff3, percentage = calc_diff(self.detected3, self.real)
            print("Tip Version 3 was %s Percent correct." % (percentage * 100))
            diff4, percentage = calc_diff(self.detected4, self.real)
            print("Tip Version 4 was %s Percent correct." % (percentage * 100))
            diff5, percentage = calc_diff(self.detected5, self.real)
            print("Tip Version 5 was %s Percent correct." % (percentage * 100))
            datas = zip(self.detected, self.detected2, self.detected3, self.detected4, self.detected5, self.real,
                        self.was_covert, self.frame_no, diff, diff2, diff3, diff4, diff5)
            for each in datas:
                entry = list(each)
                # if each[0] == each[1]:
                #     entry.append(True)
                # else:
                #     entry.append(False)
                spamwriter.writerow(entry)

    def __init__(self, inp):
        if isinstance(inp, str):
            self.fname = inp.split('.')[0]
        self.camera = Camera(device=inp, output=self.fname)
        self.board = Board()
        camconf = "camera_config.json"
        baord_conf = "boardconfig.json"
        if os.path.isfile(camconf):
            self.camera.load_config(filename=camconf)
        else:
            self.camera.do_calibration(img=True)
            self.camera.save_config(camconf)
        if self.board_config_load and os.path.isfile(baord_conf):
            with open(baord_conf, 'r') as bc:
                imgps = json.loads(bc.readline())

            self.Calibrated = BoardCalibrator(camera=self.camera, imgpts=imgps, board=self.board)
        else:
            self.Calibrated = BoardCalibrator(camera=self.camera)
            with open("boardconfig.json", 'w') as bc:
                imgps = self.Calibrated.imgpoints
                bc.write(json.dumps(imgps))
        self.Substractor = BackgroundSubtractor(c1=inp, camera=self.camera)
        plt.ion()
        self.figure = plt.figure()
        self.plt1 = self.figure.add_subplot(111)
        self.line1, = self.plt1.plot(range(200), [0] * 200, 'r.-')
        self.plt1.axis([0, 200, 0, 10000])
        cv2.namedWindow("Current", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Current", 20, 20)
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Original", 20, 500)
        cv2.namedWindow("Points", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Current", 1000, 20)
        cv2.namedWindow("Blobimg", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Points", self._click)
        mixer.init()
        # mixer.music.load('beep.mp3')
        # cv2.namedWindow("FG Substraction", cv2.WINDOW_NORMAL)
        # cv2.createTrackbar("History", "Current", self.history, 1000, self._set_history)
        # cv2.createTrackbar("Shadow Treshold", "Current", int(self.shad_tresh * 100), 100, self._set_shad_tresh)
        # cv2.createTrackbar("VarThreshold", "Current", self.var_tresh, 100, self._set_var_tresh)
        # cv2.createTrackbar("VarMax", "Current", self.var_max, 100, self._set_var_max)
        # cv2.createTrackbar("VarMin", "Current", self.var_min, 100, self._set_var_min)

        self.Substractor.start()

        realboard = np.zeros((self.camera.height, self.camera.width, 3), np.uint8)
        # self.frame = self.camera.undistort_image(img)
        for i in self.Calibrated.imp:
            try:
                realboard[i[0][1], i[0][0]] = [0, 0, 255]
            except IndexError:
                pass

        added = 0
        while True:
            img = self.Substractor.get_image()
            if img is not None:

                img = cv2.add(realboard, img)
                cv2.imshow("Original", img)
                cv2.imshow("Points", self.board.draw_board())
                if self.Substractor.stopped:
                    self.write_data()
                    exit()
                cv2.imshow("Current", self.Substractor.get_substracted())
                storage, unaltered = self.Substractor.get_storage()
                y = [x[2] for x in storage]
                y = unaltered
                self.line1.set_xdata(range(len(y)))
                self.line1.set_ydata(y)
                k = cv2.waitKey(1)
                if k == ord('a'):
                    self.add_dart(frame_no=self.camera.read_frame_no)
                if k == ord('s'):
                    self.write_data()
                if k == ord('w'):
                    pass
                    self.figure.savefig(r"thesisimages/plot.jpg")
                if k == ord('f'):
                    added = 0
                    self.Substractor.clear_arrows()
                if k == 27:
                    self.Substractor.stopped = True
                    break
                if k == 119:
                    print("Pressed w Key so Waiting")
                    cv2.waitKey(-1)
            arrows = self.Substractor.get_arrows()
            i = 1
            for each in arrows:
                tip = each.tip
                frame_no = each.frame_no
                points = self.Calibrated.calculate_points(tip)
                if i > added:
                    self.add_dart(arrow=each, detected=points, frame_no=frame_no)
                    added += 1
                i += 1
            if added >= 3:
                added = 0
                self.Substractor.clear_arrows()
        self.write_data()

    def _click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            nx = x - self.board.size / 2
            ny = self.board.size / 2 - y
            print("Clicked %s " % self.board.calculate_field([[nx], [ny]]))
            self.imgpoint = [x, y]
            self.boardpoint = [[nx], [ny]]
            # self.imgpoints.append([x, y])

    def add_dart(self, arrow=None, detected="N/D", frame_no=0):
        self.Substractor.paused = True
        # mixer.music.play()
        if arrow is not None:
            print("Ratio is %s" % arrow.ratio)
            print([cv2.contourArea(x) for x in arrow.contours])
            points = self.Calibrated.calculate_points(arrow.tip)
            pimg = self.board.draw_field(points)
            cv2.imshow("Points", pimg)
            cv2.imshow("Blobimg", arrow.img)
            k = -1
            while k not in [13, 32] or self.boardpoint is None:
                k = cv2.waitKey(-1)
            if k == 13:
                print("Enter")
                print(len(self.detected))
                self.was_covert.append(False)
            if k == 32:
                self.was_covert.append(True)
            self.real.append(self.board.calculate_field(self.boardpoint))
            self.boardpoint = None
        # print("Adding an arrow:")
        else:
            k = -1
            while k not in [13, 32] or self.boardpoint is None:
                k = cv2.waitKey(-1)
            if k == 13:
                print("Enter")
                print(len(self.detected))
                self.was_covert.append(False)
            if k == 32:
                self.was_covert.append(True)
            self.real.append(self.board.calculate_field(self.boardpoint))
        self.frame_no.append(frame_no)
        # inp = raw_input("What were the real Points? Type 'n' if the dart is not at the board: ")
        # covert = raw_input("Was the arrow covert by another one? 'n' for no, 'y' for yes: ")
        # if inp == 'n':
        #     self.real.append('N/D')
        # else:
        #     self.real.append(inp)
        if arrow is not None:
            points = self.Calibrated.calculate_points(arrow.tip)
            print(points)
            self.detected.append(points)
            points = self.Calibrated.calculate_points(arrow.tip2)
            print(points)
            self.detected2.append(points)
            points = self.Calibrated.calculate_points(arrow.tip3)
            print(points)
            self.detected3.append(points)
            points = self.Calibrated.calculate_points(arrow.tip4)
            print(points)
            self.detected4.append(points)
            points = self.Calibrated.calculate_points(arrow.tip5)
            print(points)
            self.detected5.append(points)
        else:
            self.detected.append("N/D")
            self.detected2.append("N/D")
            self.detected3.append("N/D")
            self.detected4.append("N/D")
            self.detected5.append("N/D")
        # if covert == 'y':
        #     self.was_covert.append(True)
        # else:
        #     self.was_covert.append(False)
        self.Substractor.paused = False


class BoardCalibrator(object):
    imgpoints = []

    def __init__(self, input=0, camera=None, imgpts=None, board=None):
        if not isinstance(camera, Camera):
            camera = Camera(device=input)

        self.camera = camera
        self.frame, reseted = self.camera.get_image()

        if board is None:
            self.board = Board()
        else:
            self.board = board
        allobj = self.board._get_configs()
        [i.append(0) for i in allobj]
        nobj = np.array(allobj, np.float64)
        if imgpts is None:
            cv2.namedWindow("Calibration Window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Calibration Hint", cv2.WINDOW_NORMAL)
            cv2.imshow("Calibration Window", self.frame)
            cv2.waitKey(1)
            config_points = nobj
            cv2.setMouseCallback("Calibration Window", self.click)
            pos = 0
            for i in config_points:
                print("Select field %s please. Accept with any key." % str(i))
                hintim = self.board.get_config_hint(pos)
                cv2.imshow("Calibration Hint", hintim)
                k = cv2.waitKey(-1) & 0xFF
                self.imgpoints.append(self.imgpoint)
                if k == 27:
                    print("Escaped and closing.")
                    break
                else:
                    print("Thank you")
                    pos += 1

        else:
            self.imgpoints = imgpts

        print("Imagepoints %s" % self.imgpoints)
        print("Objp %s" % nobj)

        print(self.camera.config['dist'])
        print(np.array(self.camera.config['mtx']))
        _, rvec, tvec = cv2.solvePnP(nobj,
                                     np.array(self.imgpoints, np.float64), np.array(self.camera.config['mtx']),
                                     np.array(self.camera.config['dist']), None, None, False, cv2.SOLVEPNP_ITERATIVE)
        mean_error = 0
        imgpoints2, _ = cv2.projectPoints(nobj, rvec, tvec, self.camera.cameramatrix, self.camera.config['dist'])
        impoints = np.array([[x] for x in self.imgpoints], np.float64)
        error = cv2.norm(impoints, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

        print("total error: ", mean_error)
        outers = self.board.get_radius()

        def calcy(x, r):
            return np.sqrt(np.power(r, 2) - np.power(x, 2))

        points = []
        for outer in outers:
            xs = [i * 0.1 for i in range(0, int(outer * 10))]
            xs += [i * -0.1 for i in range(0, int(outer * 10))]
            for i in xs:
                y = calcy(i, int(outer))
                points.append([[i], [y], [0]])
                points.append([[i], [-y], [0]])
        points = np.array(points)
        # points = np.array([[[0],[0],[0]], [[outer],[0],[0]]])
        imp, jac = cv2.projectPoints(points, rvec, tvec, np.array(self.camera.config['mtx']),
                                     np.array(self.camera.config['dist']))
        rot, _ = cv2.Rodrigues(rvec)
        rev = projectReverse(self.imgpoints, rot, tvec, self.camera.config['mtx'])
        self.rot = rot
        self.tvec = tvec
        # b.draw_board_to_frame(frame2)
        print("Imagepoints %s" % self.imgpoints)
        print("Objecpoints %s" % rev)
        cv2.destroyAllWindows()
        self.imp = imp
        # cv2.setMouseCallback("Calibration Window", self._calcObj)
        # while True:
        #     self.frame,reseted  = self.camera.get_image()
        #     self.frame = self.camera.undistort_image(self.frame)
        #     for i in imp:
        #         try:
        #             self.frame[i[0][1], i[0][0]] = [0,0,255]
        #         except IndexError:
        #             pass
        #     # self.frame = cv2.circle(self.frame,(int(np.round(imp[0][0][0])), int(np.round(imp[0][0][1]))),int(np.round(dist)),[0,0,255],1)
        #     cv2.imshow("Calibration Window", self.frame)
        #     k = cv2.waitKey(1) & 0xFF
        #     if k == 27:
        #         break
        #
        # k = cv2.waitKey(-1) & 0xFF

    def calculate_points(self, point):
        x, y = point
        objp = projectReverse(np.array([[[x], [y]]]), self.rot, self.tvec, self.camera.config['mtx'])
        return self.board.calculate_field(objp[0][:2])

    def _calcObj(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked %s: %s" % (x, y))

            objp = projectReverse(np.array([[[x], [y]]]), self.rot, self.tvec, self.camera.config['mtx'])
            print("Object Point %s" % objp)
            print("Field: %s" % self.board.calculate_field(objp[0][:2]))

    def click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked %s: %s" % (x, y))
            self.imgpoint = [x, y]
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
        print('utils.py -i <inputputfile>')
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
            ma = MainApplikacation(inputfile)
            # camera = Camera(inputfile)
            # bc = BoardCalibrator(inputfile)
            # bs = BackgroundSubtractor(inputfile)
            # bd = BlobDetector(c1)

        elif opt in ("-d", "--device"):
            if arg == "":
                device = 0
            device = int(arg)

            print("With device called")
            # c1 = cv2.VideoCapture(1)
            # c1.set(3, width)
            # c1.set(4, height)
            # able_to_read, f1 = c1.read()
            # hsv = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
            # print(able_to_read)
            # cc = CountourDetector(c1)
            ma = MainApplikacation(device)
            # camera = Camera(inputfile)
            # bc = BoardCalibrator(device)
            # bs = BackgroundSubtractor(device)
            # bd = BlobDetector(c1)
            # bc = BoardCalibrator(device)
    print('Output file is "', inputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
