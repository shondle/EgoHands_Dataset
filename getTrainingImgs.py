from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
import cv2

## these functions are called from dataset.py
## to create the PyTorch database object.

# getTrainingImage returns the colored image
# from video and frame number specified
def getTrainingImage(videoNumber, frameNumber, videos):
    videoNumber = videoNumber - 1
    frameNumber = frameNumber - 1
    img = cv2.imread(str(getFramePath(videos.iloc[videoNumber], frameNumber)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# getTrainingMask returns the black and white segmentation
# mask from video and frame number specified
def getTrainingMask(videoNumber, frameNumber, videos):
    videoNumber = videoNumber - 1
    frameNumber = frameNumber - 1

    hand_mask = getSegmentationMask(videos.iloc[videoNumber], frameNumber, 'all')

    return hand_mask
