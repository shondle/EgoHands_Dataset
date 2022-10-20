from getMetaBy import getMetaBy
from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
import scipy as sio
import cv2
import tkinter
import pandas as pd

videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
# print(str(getFramePath(videos.iloc[0], 7)))
img = cv2.imread(str(getFramePath(videos.iloc[0], 7)))
# cv2.imshow('hi', img)
# cv2.waitKey(0)

hand_mask = getSegmentationMask(videos.iloc[0], 7, 'all')

root = tkinter.Tk()
height = root.winfo_screenheight()
width = root.winfo_screenwidth()





# print(labelled_frames)
# print(((videos.iloc[0]).loc['video_id']))

## print(((videos.iloc[0]).loc['labelled_frames'])[0][7][0][0][0])

## this is how to get loc of my right
# print(((videos.iloc[0]).loc['labelled_frames'])[0][5][2])

