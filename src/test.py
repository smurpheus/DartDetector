import numpy as np
import cv2
from utils import Camera
import utils
c = Camera(device=False)
c.do_calibration(True)
utils.test_image(c)


import utils
b = utils.Board()
im = b.get_config_hint()
import cv2
cv2.imshow("asd",im)
cv2.waitKey(0)
