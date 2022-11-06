from getMetaBy import getMetaBy
from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
import numpy as np
import cv2
from matplotlib import pyplot as plt

def getTrainingImage(videoNumber, frameNumber, videos):
    videoNumber = videoNumber - 1
    frameNumber = frameNumber - 1
    img = cv2.imread(str(getFramePath(videos.iloc[videoNumber], frameNumber)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # fig = plt.figure(figsize=(4, 7))
    # fig.add_subplot(1, 1, 1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("Hand Segmentation")
    # plt.show()
    return img

def getTrainingMask(videoNumber, frameNumber, videos):
    videoNumber = videoNumber - 1
    frameNumber = frameNumber - 1

    hand_mask = getSegmentationMask(videos.iloc[videoNumber], frameNumber, 'all')

    # fig = plt.figure(figsize=(4, 7))
    # fig.add_subplot(1, 1, 1)
    # plt.imshow(hand_mask)
    # plt.axis('off')
    # plt.title("Hand Segmentation")
    # plt.show()

    return hand_mask

# videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
# fig = plt.figure(figsize=(4, 7))
# fig.add_subplot(1, 1, 1)
# plt.imshow(getTrainingMask(1, 8, videos))
# plt.axis('off')
# plt.title("Hand Segmentation")
# plt.show()
