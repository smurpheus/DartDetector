import numpy as np
import cv2
from utils import Camera
import utils
# c = Camera(device=False)
# c.do_calibration(True)
# utils.test_image(c)


import utils
import cv2
size = 800
rad = size/2 * 0.9
b = utils.Board(radius=rad)
def print_all(b):
    im = b.get_config_hint()
    width, height = im.shape[:2]
    im_map = {}
    for each in b.fields_in_order:
        im_map[each] = []
    im_map[0] = []
    for x in range(height):
        nx = x -size/2
        for y in range(width):
            ny = size/2 - y
            # dist_from_mid = np.sqrt(np.power(x, 2) + np.power(y, 2))
            # if dist_from_mid <= rad:
            field = b.calculate_field([[nx],[ny]])
            if field in im_map.keys():
                im_map[field].append([x,y])
            else:
                im_map[field] = [[x,y]]
    for each, points in im_map.iteritems():
        im = b.get_config_hint()
        for p in points:
            im[p[1],p[0]] = [255,255,0]
        print(each)
        cv2.imshow("asd",im)
        cv2.waitKey(1000)
print_all(b)