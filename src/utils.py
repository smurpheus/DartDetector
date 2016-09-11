#!/usr/bin/python
# import numpy as np
import cv2
import sys, getopt


def save_vid(fname="Deafaultoutput", size=(640, 480)):
    fname += '.avi'
    width = size[0]
    height = size[1]
    print("%s x %s"%(width,height))
    c1 = cv2.VideoCapture(1)
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