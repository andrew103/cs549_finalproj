import cv2
import yaml
import numpy as np
# import blob
import random
from surf_object_detection import surf_detect
from os.path import basename, dirname, splitext, join, isfile
import os

# Define IOU calculator
def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# Load in a list of classes
classes = ['pedestrianCrossing', 'speedLimit', 'stop', 'yield']
class2num = dict()
class2num['pedestrianCrossing'] = 5
class2num['speedLimit'] = 6
class2num['stop'] = 7
class2num['yield'] = 8
print('Classes:')
print(classes)

num2class = dict()
num2class[5] = 'pedestrianCrossing'
num2class[6] = 'speedLimit'
num2class[7] = 'stop'
num2class[8] = 'yield'

# Load in test images
print('Loading in test images...')
# images = blob.blob('../darknet2/images/val/*')

cwd = os.getcwd()
images = [join(cwd, "darknet2/images/val/", f) for f in os.listdir("darknet2/images/val")]

random.shuffle(images)
test_images = images

# For each class, choose one random image as template image
class_metrics = dict()
for c in classes:
    print(f'Processing images for class: {c}')
    template = cv2.imread(f'code/templates/{c}.jpg')

    # Run surf on each sample image.
    num_tp = 0
    num_detected = 0
    for sample in test_images:
        filename = basename(sample)
        directory = dirname(sample)
        name, ext = splitext(filename)

        # Load label
        label_path = f'{directory.replace("images","labels")}/{name}.txt'
        with open(label_path,'r') as f:
            labels = f.readlines()

        # Load image
        sample_image = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)

        # Run surf
        detected_box = surf_detect(template, sample_image)

        # sample image info
        height, width = sample_image.shape

        # Look through labels to see if class exists
        for label in labels:
            fields = label.strip().split(' ')
            if int(fields[0]) == class2num[c]:
                num_tp += 1
                box = fields[1:]

                if None in detected_box:
                    continue

                box_width = float(box[2]) * width
                box_height = float(box[3]) * height
                xa = float(box[0]) * width - box_width / 2
                ya = float(box[1]) * height - box_height / 2
                xb = xa + box_width
                yb = ya + box_height

                iou = calc_iou(detected_box, [xa, ya, xb, yb])
                # print(detected_box)
                # print(xa, ya, xb, yb)
                # print(iou)
                if (iou > 0.5): num_detected += 1
                # break
                
    # break
    # Calculate precision
    if num_tp == 0: precision = 0
    else: precision = float(num_detected) / num_tp
    class_metrics[c] = precision

print("Finished.")
print(class_metrics)