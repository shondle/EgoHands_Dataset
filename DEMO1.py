from getMetaBy import getMetaBy
from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
import scipy as sio
import cv2
import pandas as pd

videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
img = cv2.imread(str(getFramePath(videos.iloc[0], 7)))

hand_mask = getSegmentationMask(videos.iloc[0], 7, 'all')

cv2.imshow('hi', hand_mask)
cv2.waitKey(0)

