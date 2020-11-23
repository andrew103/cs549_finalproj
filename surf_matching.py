import cv2
import numpy as np
from numpy.core.defchararray import lower, upper

# import the template and test images in grayscale
template_img = cv2.imread('stop_sign_template.jpg', cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread('stop_sign.png', cv2.IMREAD_GRAYSCALE)

# use SURF to get features for both the template and test images
surf_detector = cv2.xfeatures2d.SURF_create(1000)
template_keypoints, template_descriptors = surf_detector.detectAndCompute(template_img, None)
test_keypoints, test_descriptors = surf_detector.detectAndCompute(test_img, None)

# use FLANN to match keypoints of the template and test image
flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = flann_matcher.knnMatch(template_descriptors, test_descriptors, 2)

# filter the knn matched keypoints further with a distance filter
ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# get (x, y) coordinates of matches keypoints
matched_keypoints = []
for match in good_matches:
    img_idx = match.queryIdx
    (x, y) = test_keypoints[img_idx].pt
    matched_keypoints.append((x, y))

# get limits to draw bounding box
upper_bound = None
lower_bound = None
right_bound = None
left_bound = None
for keypoint in matched_keypoints:
    if not upper_bound or upper_bound >= keypoint[1]:
        upper_bound = keypoint[1]
    if not lower_bound or lower_bound <= keypoint[1]:
        lower_bound = keypoint[1]
    if not right_bound or right_bound <= keypoint[0]:
        right_bound = keypoint[0]
    if not left_bound or left_bound >= keypoint[0]:
        left_bound = keypoint[0]

# use the bounds to draw a rectangular bounding box on the test image
corners = [(left_bound, upper_bound), (left_bound, lower_bound), (right_bound, lower_bound), (right_bound, upper_bound)]
cv2.drawContours(test_img, [np.array(corners).reshape((-1,1,2)).astype(np.int32)], 0, (0, 255, 0), 2)
cv2.imshow('Bounding Box', test_img)
# cv2.waitKey()

# draw the matched features in the images side-by-side
# print(good_matches)
img_matches = np.empty((max(template_img.shape[0], test_img.shape[0]), template_img.shape[1]+test_img.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(template_img, template_keypoints, test_img, test_keypoints, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matches', img_matches)
cv2.waitKey()