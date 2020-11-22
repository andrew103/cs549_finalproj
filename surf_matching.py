import cv2
import numpy as np

template_img = cv2.imread('stop_sign_template.jpg', cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread('stop_sign.png', cv2.IMREAD_GRAYSCALE)

surf_detector = cv2.xfeatures2d.SURF_create(1000)
template_keypoints, template_descriptors = surf_detector.detectAndCompute(template_img, None)
test_keypoints, test_descriptors = surf_detector.detectAndCompute(test_img, None)

flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = flann_matcher.knnMatch(template_descriptors, test_descriptors, 2)

ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty((max(template_img.shape[0], test_img.shape[0]), template_img.shape[1]+test_img.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(template_img, template_keypoints, test_img, test_keypoints, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matches', img_matches)
cv2.waitKey()