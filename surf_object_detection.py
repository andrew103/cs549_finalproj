import cv2
import numpy as np
from numpy.core.defchararray import lower, upper

def remove_outliers(keypoints, thresh = 2):
    num_points = len(keypoints)
    if num_points == 0:
        return keypoints
    avgx = 0
    avgy = 0
    for kp in keypoints:
        avgx += kp[0]
        avgy += kp[1]
    avgx /= num_points
    avgy /= num_points
    avg_dist = 0
    for kp in keypoints:
        dist = np.sqrt((kp[0] - avgx)**2 + (kp[1] - avgy)**2)
        avg_dist += dist
    avg_dist /= num_points

    points = []
    for kp in keypoints:
        dist = np.sqrt((kp[0] - avgx)**2 + (kp[1] - avgy)**2)
        if (dist < avg_dist * thresh): points.append(kp)
    return points

def surf_detect(template_img, test_img):

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

    good_matches = sorted(good_matches, key=lambda x:x.distance)[:20]

    # get (x, y) coordinates of matches keypoints
    matched_keypoints = []
    for match in good_matches:
        img_idx = match.trainIdx
        (x, y) = test_keypoints[img_idx].pt
        matched_keypoints.append((x, y))

    matched_keypoints = remove_outliers(matched_keypoints)

    # matched_keypoints = sorted(matched_keypoints, key=lambda x:x[2])

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
    corners = [left_bound, upper_bound, right_bound, lower_bound]
    return corners