# import cv2
# import numpy as np
#
# # mouse callback function
# from src.utils import Camera
#
#
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#
# # Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# # cv2.namedWindow('image')
# # cv2.setMouseCallback('image',draw_circle)
# cam = Camera(device=0, save_file=False)
# cam.do_calibration()
# while(1):
#     img, readable = cam.get_image()
#     if not readable:
#         cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()