import numpy as np
import cv2
from utils import Camera
import utils
c = Camera(device=False)
c.do_calibration(True)
utils.test_image(c)