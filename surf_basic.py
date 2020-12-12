# https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html

import cv2
import matplotlib as plt

img = cv2.imread("stop_sign_testing/stop_sign_template.jpg", 0)
surf = cv2.xfeatures2d.SURF_create(1000)
# surf.setUpright(True)

kp, des = surf.detectAndCompute(img, None)

img_surf = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)

cv2.imshow('Image Features', img_surf)
cv2.waitKey()