from getMetaBy import getMetaBy
from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
import scipy as sio
import cv2
import pandas as pd

videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
img = cv2.imread(str(getFramePath(videos.iloc[0], 7)))

hand_mask = getSegmentationMask(videos.iloc[0], 7, 'all')
bounding_boxes = getBoundingBoxes(videos.iloc[0], 7)

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.imshow('window', hand_mask)
cv2.waitKey(0)

# testing rectangles from getBoundingBoxes
rect = cv2.rectangle(img, np.int32(bounding_boxes[0]), (255, 255, 255), 3)
rect = cv2.rectangle(img, np.int32(bounding_boxes[1]), (255, 255, 255), 3)
rect = cv2.rectangle(img, np.int32(bounding_boxes[2]), (255, 255, 255), 3)
rect = cv2.rectangle(img, np.int32(bounding_boxes[3]), (255, 255, 255), 3)

cv2.imshow('window', rect)
cv2.waitKey(0)
