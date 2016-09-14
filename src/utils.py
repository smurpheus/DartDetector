#!/usr/bin/python
import numpy as np
import cv2
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


def calibration_pictures(size=(640, 480), device=0):
    width = size[0]
    height = size[1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_h * chess_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)
    print("%s x %s" % (width, height))
    c1 = cv2.VideoCapture(device)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for i in range(14):
        print ("Select field %s please. Accept with any key." % (self.fields.keys()[i]))
        k = cv2.waitKey(-1) & 0xFF
        if k == 27:
            print("Escaped and closing.")
            break
        else:
            print("Thank you")
        ret, corners = False, None
        readable, frame = c1.read()
        if readable:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
        while(not ret):
            readable, frame = c1.read()
            if readable:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (chess_w, chess_h), corners2, ret)
        cv2.imshow('img', frame)
    print("Objpoints %s"%objpoints)
    print("Imgpoints %s" % imgpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("MTX %s"%mtx)
    print("DIST %s" % dist)
    print("rvecs %s" % rvecs)
    print("tvecs %s" % tvecs)

def main(argv):
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:s:",["ofile=", "size="])
    except getopt.GetoptError:
        print 'utils.py -o <outputfile>'
        sys.exit(2)
    width = 640
    height = 480
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

    print 'Output file is "', outputfile
    save_vid(outputfile, (width, height))

if __name__ == "__main__":
   main(sys.argv[1:])