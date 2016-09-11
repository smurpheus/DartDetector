import numpy as np
import cv2
height = 480
width = 640


def save_vid(fname="Deafaultoutput"):
    fname += '.avi'
    c1 = cv2.VideoCapture(1)
    c1.set(3, height)
    c1.set(4, width)
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