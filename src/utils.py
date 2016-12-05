#!/usr/bin/python
import numpy as np
import cv2
import json
import sys, getopt
from pygame import mixer
import glob
from operator import itemgetter
import time
import matplotlib.pyplot as plt
from collections import deque
import timeit
from matplotlib import pyplot as plt

chess_w = 9
chess_h = 6
board = [170. / 170., 162. / 170., 107. / 170., 99. / 170., 15.9 / 170., 6.35 / 170.]


class Board:
    circles = [170. / 170., 162. / 170., 107. / 170., 99. / 170., 15.9 / 170., 6.35 / 170.]
    circles = [170. / 170., 160. / 170., 107. / 170., 97. / 170., 15.9 / 170., 6 / 170.]
    angles = [i * 18 + 9 for i in range(20)]
    fields_in_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    def get_config_hint(self, cur_point=-1):
        img = np.zeros((512, 512, 3), np.uint8)
        mid = int(512 / 2.)
        rad = mid * 0.9
        for c in self.circles:
            cv2.circle(img, (mid, mid), int(c * rad), (0, 0, 255), 1)
        corners = [
            [np.sin(np.radians(i)) * rad + mid, np.cos(np.radians(i)) * rad + mid]
            for i in self.angles]
        corners2 = [
            [np.sin(np.radians(i)) * (rad * self.circles[-2]) + mid,
             np.cos(np.radians(i)) * (rad * self.circles[-2]) + mid]
            for i in self.angles]
        for corner in corners:
            x, y = corner
            x2, y2 = corners2[corners.index(corner)]
            cv2.line(img, (int(x2), int(y2)), (int(x), int(y)), (0, 0, 255), 1)
        if cur_point == -1:
            for x, y in self._get_configs(custum_rad=rad):
                x += mid
                y = (y * (-1)) + mid
                img[int(y), int(x)] = (255, 0, 0)
                cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 1)
        else:
            x, y = self._get_configs(custum_rad=rad)[cur_point]
            x += mid
            y = (y * (-1)) + mid
            img[int(y), int(x)] = (255, 0, 0)
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 1)
        return img

    def __init__(self, radius=170, center=(0, 0)):
        self.radius = radius
        self.center = center

    def get_radius(self):
        return [i * self.radius for i in self.circles]

    def get_corners(self):
        return [
            [np.sin(np.radians(i)) * self.radius + self.center[0], np.cos(np.radians(i)) * self.radius + self.center[1]]
            for i in self.angles]

    def _get_configs(self, custum_rad=False):
        if not custum_rad:
            c_radius = self.radius
        else:
            c_radius = custum_rad
        result = []
        result.append([self.center[0], self.center[1]])
        # Outer Ring
        # angle = self.angles[0]
        # radius = self.circles[0] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        # angle = self.angles[5]
        # radius = self.circles[0] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        # angle = self.angles[10]
        # radius = self.circles[0] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        # angle = self.angles[15]
        # radius = self.circles[0] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        #
        # #Inner Ring
        # angle = self.angles[2]
        # radius = self.circles[3] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        # angle = self.angles[7]
        # radius = self.circles[3] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        # angle = self.angles[12]
        # radius = self.circles[3] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        # angle = self.angles[17]
        # radius = self.circles[3] * c_radius
        # result.append([np.sin(np.radians(angle)) * radius + self.center[0],
        #                np.cos(np.radians(angle)) * radius + self.center[1]])
        angle = self.angles[3]
        radius = self.circles[1] * c_radius
        result.append([np.sin(np.radians(angle)) * radius + self.center[0],
                       np.cos(np.radians(angle)) * radius + self.center[1]])
        angle = self.angles[9]
        radius = self.circles[1] * c_radius
        result.append([np.sin(np.radians(angle)) * radius + self.center[0],
                       np.cos(np.radians(angle)) * radius + self.center[1]])
        angle = self.angles[17]
        radius = self.circles[1] * c_radius
        result.append([np.sin(np.radians(angle)) * radius + self.center[0],
                       np.cos(np.radians(angle)) * radius + self.center[1]])

        return result

    def draw_board_to_frame(self, frame):
        # Create a black image

        # Draw a blue line with thickness of 5 px
        for rad in self.get_radius():
            cv2.circle(frame, (int(self.center[0]), int(self.center[1])), int(rad), [0, 0, 255])
        for corner in self.get_corners():
            cv2.line(frame, (int(self.center[0]), int(self.center[1])), (int(corner[0]), int(corner[1])), [0, 0, 255],
                     1)
        # Display the image
        return frame

    def calculate_field(self, point):
        fields_in_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20]
        angles = [0] + [i * 18 + 9 for i in range(21)]
        multiplier = [2, 1, 3, 1.0, "25", "50"]
        x = point[0][0]
        y = point[1][0]
        dist_from_mid = np.sqrt(np.power(x, 2) + np.power(y, 2))
        angle = np.arcsin(x / dist_from_mid)
        angle = np.degrees(angle)
        if y > 0:
            if angle < 0:
                angle = 360 - abs(angle)
        else:
            if angle > 0:
                angle = 180 - abs(angle)
            else:
                angle = 180 + abs(angle)
        indexof = 1
        for i in angles:
            if angle > i and angle < angles[angles.index(i) + 1]:
                indexof = angles.index(i)
                break

        radiuses = self.get_radius() + [0]
        mult = -1
        if dist_from_mid > radiuses[0]:
            mult = False
        else:
            for i in multiplier:
                if dist_from_mid < radiuses[multiplier.index(i)] and dist_from_mid > radiuses[multiplier.index(i) + 1]:
                    mult = i
                    break
        if isinstance(mult, str):
            return mult
        else:
            return abs(mult) * fields_in_order[indexof]


def save_vid(fname="Deafaultoutput", size=(640, 480), device=0):
    fname += '.avi'
    width = size[0]
    height = size[1]
    print("%s x %s" % (width, height))
    print("opening device %s" % device)
    c1 = cv2.VideoCapture(device)
    c1.set(3, width)
    c1.set(4, height)
    ret, frame = c1.read()
    cv2.imshow('Preview', frame)
    cv2.waitKey(-1)
    fps = c1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    try:
        print("Trying to open: %s" % c1.isOpened())
        while (c1.isOpened()):
            ret, frame = c1.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
                # if len(lines) > 0:
                #     for rho, theta in lines[0]:
                #         a = np.cos(theta)
                #         b = np.sin(theta)
                #         x0 = a * rho
                #         y0 = b * rho
                #         x1 = int(x0 + 1000 * (-b))
                #         y1 = int(y0 + 1000 * (a))
                #         x2 = int(x0 - 1000 * (-b))
                #         y2 = int(y0 - 1000 * (a))
                #
                #         cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # write the flipped frame
                out.write(frame)
                cv2.imshow('Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    except KeyboardInterrupt:
        pass
    finally:
        c1.release()
        out.release()
        cv2.destroyAllWindows()


def draw_circles():
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Draw a blue line with thickness of 5 px
    for i in board:
        cv2.circle(img, (256, 256), int(170 * i), [0, 0, 255])
    # Display the image
    cv2.imshow("img", img)

    cv2.waitKey(0)


class Arrow:
    centroid = None
    tip = None
    tip2 = None
    contours = None
    img = None
    line = None
    aproximated = None
    ratio = None
    bbox = None
    min_cnt_size = 3000
    min_ratio = 2
    max_ratio = 3.5
    success = False

    def __init__(self, contours, img):
        self.contours = [x for x in contours if cv2.contourArea(x) > self.min_cnt_size]
        self.img = img
        self.detect_arrow()

    def __repr__(self):
        output = ""
        output += "Centroid=%s;;tip=%s;;Contour_num=%s;;Contour_sizes=%s;;ratio=%s;;success=%s" % (
        self.centroid, self.tip, len(self.contours), [cv2.contourArea(x) for x in self.contours], self.ratio,self.success)
        return output

    def _get_points_on_cnt(self, cnt):
        cimg = np.zeros_like(self.img)
        cv2.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        pts = zip(pts[1], pts[0])
        return pts


    def detect_arrow(self):
        ncontours = []
        for cnt in self.contours:
            if cv2.contourArea(cnt) > self.min_cnt_size:
                # calculate box around the contour
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                b = box

                dist1 = np.linalg.norm(b[0] - b[1])
                dist2 = np.linalg.norm(b[1] - b[2])
                ratio = (dist1 / dist2)
                if self.min_ratio < ratio < self.max_ratio:
                    self.bbox = box
                    self.ratio = (dist1 / dist2)
                    self.success = True
                    ncontours.append(cnt)
                    # Calculate the Moments of the contour
                    approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
                    self.aproximated = approx

                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    self.centroid = np.array([cx, cy])
                    # Calculate a fitting line trough the contour
                    rows, cols = self.img.shape[:2]
                    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)
                    self.line = (lefty, righty)

                    ##############################################
                    ###### Standard Tip ##########################
                    pts = self._get_points_on_cnt(cnt)
                    pts_on_line = []
                    for pt in pts:
                        x, y = pt
                        ly = int(np.interp(int(x), [0, cols - 1], [lefty, righty]))
                        if y < ly + 1 and y > ly - 1:
                            pts_on_line.append(pt)
                    mdist = None
                    mpt = None
                    for pt in pts_on_line:
                        p = np.array(pt)
                        dist = np.linalg.norm(self.centroid - p)
                        if mdist is None or dist > mdist:
                            mdist = dist
                            mpt = p
                    self.tip = mpt
                    # Approximate a contour

                    approx_len = cv2.arcLength(approx, True)
                    area = cv2.contourArea(cnt)
                    # calculate the possible tip
                    box = np.int0(box)
                    boxpts = self._get_points_on_cnt(box)
                    pts_on_line2 = []
                    for pt in boxpts:
                        x, y = pt
                        ly = int(np.interp(int(x), [0, cols - 1], [lefty, righty]))
                        if y < ly + 1 and y > ly - 1:
                            pts_on_line2.append(pt)
                    mdist2 = None
                    mpt2 = None
                    for pt in pts_on_line2:
                        p = np.array(pt)
                        dist = np.linalg.norm(self.centroid - p)
                        if mdist2 is None or dist > mdist2:
                            mdist2 = dist
                            mpt2 = p
                    # print "Possible tips2 %s" % (mpt2)

                    box2 = np.int0(approx)
                    approxpts2 = self._get_points_on_cnt(box2)
                    diff = np.array([np.array(x) for x in approxpts2 if x in boxpts])
                    def centeroidnp(arr):
                        length = arr.shape[0]
                        sum_x = np.sum(arr[:, 0])
                        sum_y = np.sum(arr[:, 1])
                        return [sum_x / length, sum_y / length]


                    diff2 = []
                    for pt in diff:
                        if np.linalg.norm(pt - mpt) < 50:
                            diff2.append(pt)
                    if len(diff2) > 0:

                        self.tip2 = centeroidnp(np.array(diff2))
                        # print "Possible tip2: %s "%self.tip2
                    else:
                        self.tip2 = self.tip
                    # print "Possible tips %s" % (self.tip2)
        self.contours = ncontours

class ContourStorage:
    size = 200
    storage = deque([], size)
    means = deque([0] * size, size)
    deviations = deque([0] * size, size)
    tendecy = deque([0] * size, size)
    history = 0
    percentage_of_history = 0.03
    plotting = True
    paused = False

    def __init__(self, plotting=True):
        self.plotting = plotting
        if plotting:
            plt.ion()
            self.figure = plt.figure()
            self.plt1 = self.figure.add_subplot(411)
            self.line1, = self.plt1.plot(range(self.size), [0] * self.size, 'r.-')
            self.plt2 = self.figure.add_subplot(412)
            self.line2, = self.plt2.plot(range(self.size), [0] * self.size, 'r.-')
            self.plt3 = self.figure.add_subplot(413)
            self.line3, self.line4 = self.plt3.plot(range(self.size), [0] * self.size, 'r.-', range(self.size),
                                                    [0] * self.size, 'g.-')

            # self.plt5 = self.figure.add_subplot(414)
            # self.line5, = self.plt5.plot(range(self.size), [0] * self.size, 'r.-')

    def add_to_storage(self, contours, image):
        # if len(self.storage) + 1 > self.size:
        #     self.storage.remove(self.storage[0])
        cnts = []
        acnts = []
        if not self.paused:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                barea = cv2.contourArea(box)
                if barea < 0.08 * image.size:
                    cnts.append(area)
                    acnts.append(cnt)
                else:
                    self.paused = self.history * self.percentage_of_history
                    return
            xcnt = 0
            if len(cnts) > 0:
                xcnt = max(cnts)
            self.storage.append([image, acnts, xcnt])
            if self.plotting:
                self.plot_data()
        else:
            print("Execution was paused because of a blob that was to big.")
            if self.paused <= 0:
                self.paused = False
            self.paused -= 1

    def get_arrow(self, history=500):
        blobs = self._find_blob(history)
        positions = self._find_best_contour(blobs)
        result = []
        for pos in positions:
            img, cnt, _ = self.storage[pos]
            arrow = Arrow(cnt, img)
            if arrow.success:
                result.append(arrow)
        for s, e in blobs:
            for each in range(s, e):
                self.storage[each][2] = 0
        return result

    def get_biggest_contour_image(self):
        return max(self.storage, key=itemgetter(2))

    def _calc_tendency(self):
        size = 6
        start_list = [x[2] for x in self.storage]
        mean_tendency = list(start_list)[-size:]
        if len(mean_tendency) == size:
            middle = int(len(mean_tendency) / 2)
            first_half = list(mean_tendency)[:middle]
            second_half = list(mean_tendency)[middle:]
            fm = np.mean(first_half)
            sm = np.mean(second_half)
            return sm - fm
        else:
            return 0

    def _find_best_contour(self, blobs):
        corcection_factor = 1.15
        y = [x[2] for x in self.storage]
        best = []
        for blob in blobs:
            blob_contours = y[blob[0]: blob[1]]
            blob_mean = np.mean(blob_contours)
            diffs = []
            min_num = blob_mean
            blob_mean = blob_mean * corcection_factor
            index = 0
            for blob_contour in blob_contours:
                diff = abs(blob_contour - blob_mean)
                if diff < min_num and abs(blob_contour - blob_mean) > 0:
                    min_num = diff
                    index = blob_contours.index(blob_contour)
            best.append(index + blob[0])

        return best

    def _find_blob(self, history=500):
        self.history = history
        y = [x[2] for x in self.storage]
        mean = np.mean(y)
        blobs = []
        start = False
        y3 = []
        ind = 0
        for i in y:
            if i > mean and start is False:
                start = ind
            elif i <= mean and not start is False:
                blobs.append((start, ind))
                start = False
            ind += 1

        # for i in y3:
        #     if i > 0 and start is False:
        #         start = y3.index(i)
        #     if i <= 0 and not start is False:
        #         blobs.append((start, y3.index(i)))
        #         start = False
        # blobs = [x for x in blobs if x[1]-x[0] > self.history * self.percentage_of_history]
        nblobs = []
        for blob in blobs:
            if blob[1] - blob[0] > self.history * self.percentage_of_history:
                nblobs.append(blob)
            else:
                for i in range(blob[0], blob[1]):
                    self.storage[i][2] = mean
        return nblobs

    def get_best_contours(self, history=500):
        blobs = self._find_blob(history=500)
        positions = self._find_best_contour(blobs)
        result = []
        for pos in positions:
            result += self.storage[pos][1]
        return result

    def plot_data(self):
        im, c, m = self.get_biggest_contour_image()
        y = [x[2] for x in self.storage]
        y2 = [len(x[1]) for x in self.storage]
        # y4 = []
        # for a,c,m in self.storage:
        #     for i in c:
        #         rect = cv2.minAreaRect(i)
        #         box = cv2.boxPoints(rect)
        #         dist1 = np.linalg.norm(box[0] - box[1])
        #         dist2 = np.linalg.norm(box[1] - box[2])
        #     y4.append(dist2/dist1)


        deviation = np.std(y)
        mean = np.mean(y)
        y3 = []
        for i in self.storage:
            if i[2] > mean:
                y3.append(i[2])
            else:
                y3.append(0)
        self.means.append(mean)
        self.deviations.append(deviation)
        self.plt1.axis([0, self.size, 0, max(y3) * 1.3])
        self.line1.set_xdata(range(len(y3)))
        self.line1.set_ydata(y3)

        self.plt2.axis([0, self.size, 0, max(y)])
        self.line2.set_xdata(range(len(y)))
        self.line2.set_ydata(y)

        self.plt3.axis([0, self.size, 0, max([max(self.means), max(self.deviations)])])
        self.line3.set_ydata(self.means)
        self.line4.set_ydata(self.deviations)

        # self.plt5.axis([0, self.size, 0, max(y4)])
        # self.line5.set_xdata(range(len(y4)))
        # self.line5.set_ydata(y4)
        self.figure.canvas.draw()


class Camera:
    chessboard_size = 26

    def __init__(self, width=1280, heigth=960, device=None):
        self.cameramatrix = None
        self.width = width
        self.height = heigth
        self.roi = []
        self.config = []
        self.was_configured = False
        self.device = device
        if isinstance(self.device, int):
            self.from_file = False
            self.capture = cv2.VideoCapture(self.device)
            while self.capture.isOpened() == False:
                time.sleep(1)
                print "Waiting"
            self.capture.set(3, self.width)
            self.capture.set(4, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
        elif isinstance(device, str):
            self.from_file = True
            self.capture = cv2.VideoCapture(self.device)
            tot_frame = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            print "There are %s Frames in this File" % tot_frame

            self.width = int(self.capture.get(3))
            self.heigth = int(self.capture.get(4))

    def get_image(self):
        able_to_read, f1 = self.capture.read()
        if able_to_read:
            return f1, False
        else:
            if self.from_file:
                self.capture.set(1, 0)
            able_to_read, f1 = self.capture.read()
            print "Reading Frame: ", self.capture.get(1)
            if able_to_read:
                return f1, True
            else:
                time.sleep(2)
                able_to_read, f1 = self.capture.read()
                if able_to_read:
                    return f1, True
                else:
                    False, False

    def undistort_image(self, image):
        if self.was_configured:
            # print("CONFIG \r\n%s" % self.config)
            dst = cv2.undistort(image, np.array(self.config['mtx']), np.array(self.config['dist']), None,
                                self.cameramatrix)
            # crop the image
            x, y, w, h = self.roi
            dst = dst[y:y + h, x:x + w]
            return dst
        else:
            print("THE CAMERA WAS NOT CONFIGURED!")
            return image

    def undistort_image_without_crop(self, image):
        if self.was_configured:
            dst = cv2.undistort(image, np.array(self.config['mtx']), np.array(self.config['dist']), None,
                                self.cameramatrix)

            return dst
        else:
            print("THE CAMERA WAS NOT CONFIGURED!")
            return image

    def load_config(self, filename="camera_config.json"):
        try:
            with open(filename, "r") as f:
                lines = f.read()
                print(lines)
                config = json.loads(lines)
        except:
            return False
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(config['mtx']), np.array(config['dist']),
                                                          (self.width, self.height), 1, (self.width, self.height))
        self.cameramatrix = newcameramtx
        self.roi = roi
        self.config = config
        self.was_configured = True
        return config

    def save_config(self, filename="camera_config.json"):
        if self.was_configured:
            with open(filename, "w") as f:
                f.write(json.dumps(self.config))

    def do_calibration(self, img=None):
        width = self.width
        height = self.height
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.chessboard_size, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chess_h * chess_w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)
        objp = objp * self.chessboard_size
        print("%s x %s" % (width, height))

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        cv2.namedWindow("Calibration Window", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()
        while not self.device is False and not self.capture.isOpened():
            pass
        try:
            if not img:
                for i in range(14):
                    print("Processing image %s" % i)
                    ret, corners = False, None
                    readable, frame = self.capture.read()
                    cv2.imshow('Calibration Window', frame)
                    if cv2.waitKey(-1) & 0xFF == ord('q'):
                        break
                    cv2.waitKey(5000)
                    while (not ret):
                        if img is None:
                            readable, frame = self.capture.read()
                        for _ in range(10):
                            readable, frame = self.capture.read()
                            if not readable:
                                readable = True
                                break
                        if readable:
                            cv2.imshow('Calibration Window', frame)
                            if cv2.waitKey(500) & 0xFF == ord('q'):
                                break
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
                            # if ret:
                            #     frame2 = cv2.drawChessboardCorners(frame, (chess_w, chess_h), corners, ret)
                            #     cv2.imshow('Calibration Window', frame2)
                    ret = False
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

                    # Draw and display the corners
                    cv2.imwrite("./calibraw%s.jpg" % i, frame)
                    frame = cv2.drawChessboardCorners(frame, (chess_w, chess_h), corners2, ret)
                    cv2.imwrite("./calib.jpg", frame)
                    # Load the required library

                    mixer.init()
                    mixer.music.load('beep.mp3')
                    mixer.music.play()
                    print("""##########################################################################################\r\n
                                FOUND CORNERS STILL %d!!!!!!! \r\n
                             ###########################################################################################
                          """ % (14 - i))
                    cv2.imshow('Calibration Window', frame)
                    cv2.waitKey(2000)
            else:
                images = glob.glob('./calibimgs/*.jpg')
                for fname in images:
                    frame = cv2.imread(fname)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
                    if ret == True:
                        objpoints.append(objp)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners)
                        cv2.drawChessboardCorners(frame, (chess_w, chess_h), corners2, ret)
                        cv2.imshow("Calibration Window", frame)
                        cv2.waitKey(1)
            # print("Objpoints %s" % objpoints)
            # print("Imgpoints %s" % imgpoints)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,
                                                               criteria=criteria)
            # print("MTX %s" % mtx)
            # print("DIST %s" % dist)
            # print("rvecs %s" % rvecs)
            # print("tvecs %s" % tvecs)
            mtx = mtx.tolist()
            ntvecs = []
            for i in tvecs:
                ntvecs.append(i.tolist())
            nrvecs = []
            for i in rvecs:
                nrvecs.append(i.tolist())
            nobjpoints = []
            for i in objpoints:
                nobjpoints.append(i.tolist())
            nimgpoints = []
            for i in imgpoints:
                nimgpoints.append(i.tolist())
            dist = dist.tolist()
            config = {'mtx': np.array(mtx),
                      'dist': np.array(dist),
                      'rvecs': np.array(nrvecs),
                      'tvecs': np.array(ntvecs[0]),
                      'imgpoints': nimgpoints,
                      'objpoints': nobjpoints}
            self.config = config
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(config['mtx']), np.array(config['dist']),
            # (self.width, self.height))
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(config['mtx']), np.array(config['dist']),
                                                              (self.width, self.height), 0, (self.width, self.height))
            self.cameramatrix = newcameramtx
            self.roi = roi
            print(self.cameramatrix)
            self.was_configured = True
            mean_error = 0
            for i in xrange(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], newcameramtx, np.array(dist))
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            print "total error: ", mean_error / len(objpoints)


        except KeyboardInterrupt:
            pass
        finally:
            # if not self.device is False:
            #     self.capture.release()
            cv2.destroyAllWindows()


def get_color_diffs(colors):
    diffs = []
    for i in colors:
        for e in colors[colors.index(i) + 1:]:
            diffs.append(abs(i - e))
    return diffs


def projectReverse(imgpoints, rvec, tvec, cameramatrix):
    invroto = np.linalg.inv(rvec)
    invcam = np.linalg.inv(cameramatrix)
    objpoints = []
    for point in imgpoints:
        p = np.array(point).reshape(2, 1)
        ip = np.concatenate((p, [[1]]), 0)
        tempmat = np.dot(np.dot(invroto, invcam), ip)
        tempmat2 = np.dot(invroto, tvec)
        s = 0 + tempmat2[2][0]
        a = tempmat[2][0]
        s /= a
        reverse = np.dot(invroto, np.dot(invcam, np.array(ip) * s)) - np.dot(invroto, tvec)
        objpoints.append(reverse)
    return objpoints


def test_image(camera):
    roto, _ = cv2.Rodrigues(camera.config['rvecs'][0])
    invroto = np.linalg.inv(roto)
    print "Rotation Matrix: \r\n %s" % roto
    print "InverseRotation Matrix: \r\n %s" % invroto
    tvec = camera.config['tvecs']
    print "Translation vector: \r\n %s" % tvec
    print "Camera Matrix: \r\n %s" % camera.config['mtx']
    invcam = np.linalg.inv(camera.config['mtx'])
    print "Inverse Camera Matrix: \r\n %s" % invcam
    print "dist parameter: \r\n %s" % camera.config['dist']
    newp, _ = cv2.projectPoints(np.array(camera.config['objpoints'][0]), roto, tvec, camera.config['mtx'],
                                camera.config['dist'])
    # print np.array(camera.config['imgpoints'][0])
    # for i, x in zip(newp, np.array(camera.config['imgpoints'][0])):
    #     print "%s::: %s"%(i, x)
    # print newp
    trans = np.concatenate((roto, tvec), 1)
    print "combined Trans and Rot: \r\n %s" % trans
    newp2 = []
    for i in camera.config['objpoints'][0]:
        p = np.array(i + [1]).reshape(4, 1)
        ip = np.array(np.array(i).reshape(3, 1))
        # print ip
        # newpoint =  np.dot(np.array(camera.config['mtx']), np.dot(trans, p))
        # print "NP normal: %s"%newpoint
        newpoint = np.dot(np.array(camera.config['mtx']), np.dot(roto, ip) + tvec)

        nnp = [[newpoint[0][0] / newpoint[2][0]], [newpoint[1][0] / newpoint[2][0]]]
        print "NP crazy: %s" % nnp
        # tempmat =  np.dot(np.dot(invroto, invcam), nnp)
        # tempmat2 = np.dot(invroto, tvec)
        # s = 0 + tempmat2[2][0]
        # a = tempmat[2][0]
        # s /= a

        reverse = projectReverse([nnp], roto, tvec, camera.config['mtx'])
        # reverse = np.dot(invroto,np.dot(invcam, np.array(nnp)*s)) - np.dot(invroto,tvec)
        print "reversed: %s" % reverse
        newp2.append(nnp)
    for i, x in zip(newp, newp2):
        print "%s::: %s" % (i, x)

        #     rotated = np.dot(roto, ip)
        #     # print np.add(rotated, tvec)
        #     # print np.dot(np.array(camera.config['mtx']), np.dot(trans, p))


class Test:
    param = 200
    cannyup = 150
    cannylow = 50

    def fix_val1(self, params):
        print "called"
        self.param = params

    def fix_val2(self, params):
        print "called"
        self.cannyup = params

    def fix_val3(self, params):
        print "called"
        self.cannylow = params

    def show_image_with_calib(self, size=(1280, 960), device=0):

        width = size[0]
        height = size[1]
        print("%s x %s" % (width, height))
        c1 = cv2.VideoCapture(device)
        c1.set(3, width)
        c1.set(4, height)
        with open("camera_config.json", "r") as f:
            lines = f.read()
            print(lines)
            config = json.loads(lines)
        print config
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(config['mtx']), np.array(config['dist']),
                                                          (width, height), 1, (width, height))
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("slider", "preview", 250, 500, self.fix_val1)
        cv2.createTrackbar("cannyup", "preview", 250, 500, self.fix_val2)
        cv2.createTrackbar("cannylow", "preview", 250, 500, self.fix_val3)

        try:
            while (c1.isOpened()):
                ret, frame = c1.read()
                if ret == True:
                    dst = cv2.undistort(frame, np.array(config['mtx']), np.array(config['dist']), None, newcameramtx)
                    # crop the image
                    x, y, w, h = roi
                    dst = dst[y:y + h, x:x + w]

                    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, self.cannylow, self.cannyup, apertureSize=3)
                    # minLineLength = 100
                    # maxLineGap = 10
                    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, minLineLength, maxLineGap)
                    # for x1, y1, x2, y2 in lines[0]:
                    #     cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    print self.param
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, self.param)
                    if lines is not None and len(lines) > 0:
                        for i in lines:
                            rho, theta = i[0]
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000 * (-b))
                            y1 = int(y0 + 1000 * (a))
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * (a))

                            cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)


                            # write the flipped frame
                    cv2.imshow('Original', frame)
                    cv2.imshow('preview', dst)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    c1.set(1, 0)
        except KeyboardInterrupt:
            pass
        finally:
            c1.release()
            cv2.destroyAllWindows()


def main(argv):
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "h:i:o:s:d:r:c:", ["ofile=", "size=", "device="])
    except getopt.GetoptError as e:
        print 'utils.py -o <outputfile>'
        print e
        sys.exit(2)
    width = 640
    height = 480
    device = False
    record = False
    calibrate = False
    for opt, arg in opts:
        if opt == '-h':
            print 'utils.py -o <outputfile>'
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--size"):
            size = arg
            print("Size operator called %s" % size)
            if int(size) == 1280:
                width = 1280
                height = 960
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("-r", "--record"):
            record = True
        elif opt in ("-c", "--calibrate"):
            calibrate = True

    print 'Output file is "', outputfile
    if record:
        if device:
            save_vid(outputfile, (width, height), device)
        else:
            save_vid(outputfile, (width, height))
    if calibrate:
        if device:
            pass
            # calibration_pictures((width,height),device)
        else:
            pass
            # calibration_pictures((width, height))


if __name__ == "__main__":
    main(sys.argv[1:])
