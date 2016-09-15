#!/usr/bin/python
import numpy as np
import cv2
import json
import sys, getopt
chess_w = 9
chess_h = 6
board = [170./170., 162./170., 107./170., 99./170.,15.9/170.,6.35/170.]


def save_vid(fname="Deafaultoutput", size=(640, 480), device=0):
    fname += '.avi'
    width = size[0]
    height = size[1]
    print("%s x %s"%(width,height))
    c1 = cv2.VideoCapture(device)
    c1.set(3, width)
    c1.set(4, height)
    fps = c1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    try:
        while (c1.isOpened()):
            ret, frame = c1.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
                if len(lines) > 0:
                    for rho, theta in lines[0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

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
        cv2.circle(img, (256,256),int(170*i),[0,0,255])
    # Display the image
    cv2.imshow("img", img)

    cv2.waitKey(0)

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
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(config['mtx']), np.array(config['dist']), (width, height), 1, (width, height))
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("slider", "preview", 0, 500, self.fix_val1)
        cv2.createTrackbar("cannyup", "preview", 0, 500, self.fix_val2)
        cv2.createTrackbar("cannylow", "preview", 0, 500, self.fix_val3)

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
                    break
        except KeyboardInterrupt:
            pass
        finally:
            c1.release()
            cv2.destroyAllWindows()


def calibration_pictures(size=(640, 480), device=0):
    width = size[0]
    height = size[1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_h * chess_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)
    print("%s x %s" % (width, height))
    c1 = cv2.VideoCapture(device)
    c1.set(3, width)
    c1.set(4, height)
    c1.set(cv2.CAP_PROP_FPS, 30)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    cv2.namedWindow("Calibration Window", cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    while not c1.isOpened():
        pass
    try:
        for i in range(14):
            print("Processing image %s"%i)
            ret, corners = False, None
            readable, frame = c1.read()
            # print("Tried to read %s"%readable)
            # while (c1.isOpened()):
            #     readable, frame = c1.read()
            #     if readable == True:
            #         # write the flipped frame
            #         cv2.imshow('Calibration Window', frame)
            #         if cv2.waitKey(1) & 0xFF == ord('n'):
            #             break
            #     else:
            #         break
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
            # print("Ret: %s"%ret)
            while(not ret):
                readable, frame = c1.read()
                for i in range(10):
                    readable, frame = c1.read()
                    if not readable:
                        readable = True
                        break
                if readable:
                    cv2.imshow('Calibration Window', frame)
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
                    if ret:
                        frame2 = cv2.drawChessboardCorners(frame, (chess_w, chess_h), corners, ret)
                        cv2.imshow('Calibration Window', frame2)
            ret = False
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, (chess_w, chess_h), corners2, ret)
            print("""##########################################################################################\r\n
                        FOUND CORNERS STILL %d!!!!!!! \r\n
                     ###########################################################################################
                  """%(14-i))
            cv2.imshow('Calibration Window', frame)
            cv2.waitKey(2000)
        print("Objpoints %s"%objpoints)
        print("Imgpoints %s" % imgpoints)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("MTX %s"%mtx)
        print("DIST %s" % dist)
        print("rvecs %s" % rvecs)
        print("tvecs %s" % tvecs)
        mtx = mtx.tolist()
        ntvecs = []
        for i in tvecs:
            ntvecs.append(i.tolist())
        nrvecs = []
        for i in rvecs:
            nrvecs.append(i.tolist())
        dist = dist.tolist()
        config = {'mtx': mtx,
         'dist': dist,
         'rvecs': nrvecs,
         'tvecs': ntvecs}
        with open("camera_config.json", "w") as f:
            f.write(json.dumps(config))

    except KeyboardInterrupt:
        pass
    finally:
        c1.release()
        cv2.destroyAllWindows()

def main(argv):
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"h:i:o:s:d:r:c:",["ofile=", "size=", "device="])
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
            print("Size operator called %s"%size)
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
            calibration_pictures((width,height),device)
        else:
            calibration_pictures((width, height))

if __name__ == "__main__":
   main(sys.argv[1:])