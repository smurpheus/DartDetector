#!/usr/bin/python
import numpy as np
import cv2
import json
import sys, getopt
from pygame import mixer
import glob

chess_w = 9
chess_h = 6
board = [170. / 170., 162. / 170., 107. / 170., 99. / 170., 15.9 / 170., 6.35 / 170.]
impoints = [[565, 407], [644, 90], [738, 104], [822, 148], [885, 218], [919, 317], [915, 437], [872, 555], [793, 660],
            [691, 737], [576, 772], [466, 767], [375, 731], [309, 661], [269, 576], [262, 481], [280, 386], [321, 292],
            [383, 214], [460, 147], [549, 105]]


class Board:
    circles = [170. / 170., 162. / 170., 107. / 170., 99. / 170., 15.9 / 170., 6.35 / 170.]
    circles = [170. / 170., 160. / 170., 105. / 170., 97. / 170., 14.9 / 170., 6 / 170.]
    angles = [i * 18 + 9 for i in range(20)]
    fields_in_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    def get_config_hint(self, cur_point = -1):
        img = np.zeros((512, 512, 3), np.uint8)
        mid = int(512 /2.)
        rad = mid * 0.9
        for c in self.circles:
            cv2.circle(img, (mid,mid), int(c * rad), (0,0,255), 1)
        corners = [
            [np.sin(np.radians(i)) * rad + mid, np.cos(np.radians(i)) * rad + mid]
            for i in self.angles]
        corners2 = [
            [np.sin(np.radians(i)) * (rad * self.circles[-2]) + mid, np.cos(np.radians(i)) * (rad * self.circles[-2]) + mid]
            for i in self.angles]
        for corner in corners:
            x, y = corner
            x2, y2 = corners2[corners.index(corner)]
            cv2.line(img, (int(x2),int(y2)), (int(x),int(y)),(0,0,255), 1)
        if cur_point == -1:
            for x, y in self._get_configs(custum_rad=rad):
                x += mid
                y = (y*(-1)) + mid
                img[int(y),int(x)] = (255, 0, 0)
                cv2.circle(img, (int(x),int(y)), 10, (0, 255, 0), 1)
        else:
            x,y = self._get_configs(custum_rad=rad)[cur_point]
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

    def _get_configs(self, custum_rad = False):
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
        angle = self.angles[2]
        radius = self.circles[0] * c_radius
        result.append([np.sin(np.radians(angle)) * radius + self.center[0],
                       np.cos(np.radians(angle)) * radius + self.center[1]])
        angle = self.angles[10]
        radius = self.circles[0] * c_radius
        result.append([np.sin(np.radians(angle)) * radius + self.center[0],
                       np.cos(np.radians(angle)) * radius + self.center[1]])
        angle = self.angles[15]
        radius = self.circles[0] * c_radius
        result.append([np.sin(np.radians(angle)) * radius + self.center[0],
                       np.cos(np.radians(angle)) * radius + self.center[1]])

        return  result

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
        if isinstance(device, int):
            self.from_file = False
            self.capture = cv2.VideoCapture(self.device)
            self.capture.set(3, self.width)
            self.capture.set(4, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
        elif isinstance(device, str):
            self.from_file = True
            self.capture = cv2.VideoCapture(self.device)
            self.width = int(self.capture.get(3))
            self.heigth = int(self.capture.get(4))

    def get_image(self):
        able_to_read, f1 = self.capture.read()
        if able_to_read:
            return f1
        else:
            if self.from_file:
                self.capture.set(1, 0)
            able_to_read, f1 = self.capture.read()
            if able_to_read:
                return f1

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
