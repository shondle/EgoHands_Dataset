from getMetaBy import getMetaBy
from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
from getBoundingBoxes import getBoundingBoxes
import numpy as np
import cv2
from matplotlib import pyplot as plt


# This demo shows how to load and access ground-truth data for any of the videos.

# Let's load all videos at the courtyard location where the activity was puzzle solving.
# getMetaBy() returns a struct array that contains all possible meta information (including
# the ground-truth data) about the videos. Check the getMetaBy() documentation for more.

videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')

# Each video has 100 annotated frames. Let's consider the first video. One can access the 8th frame
# of the first video like this:
videoNumber = 1 # enter which video you want here
frameNumber = 8 # enter frame number you want here

videoNumber = videoNumber - 1
frameNumber = frameNumber - 1

# creating figure to display
fig = plt.figure(figsize=(4, 7))
rows = 3
columns = 1

# getting colored image
img = cv2.imread(str(getFramePath(videos.iloc[videoNumber], frameNumber)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# showing colored image on figure
fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Video: " + (videos.iloc[videoNumber]).loc['video_id'][videoNumber] + " - Frame #%s" % (frameNumber+1))


# Here is how to get a binary mask with hand segmentations for the current frame. The third argument
# implies that the mask will show "all" hands. To get masks for specific hands, change this argument
# to e.g. "my_right" or "yours" to get only the observer's right hand or only the other actor's
# hands respectively. Check the getSegmentationMask() documentation for more.
hand_mask = getSegmentationMask(videos.iloc[videoNumber], frameNumber, 'all')


# The bounding boxes for each hand are also easily accessible. The function below returns a 4x4
# matrix, where each row corresponds to a hand bounding box in the format [x y width height], where
# x and y mark the top left corner of the box. The rows from top to bottom contain the bounding
# boxes for "own left", "own right", "other left", and "other right" hand respectively. If a hand
# is not in the frame, the values are set to 0.
bounding_boxes = getBoundingBoxes(videos.iloc[videoNumber], frameNumber)

## assigning colors to each of the bounding boxes
## Blue
rect = cv2.rectangle(img, np.int32(bounding_boxes[0]), (0, 0, 255), 3)
## Yellow
rect = cv2.rectangle(img, np.int32(bounding_boxes[1]), (255, 255, 0), 3)
## Red
rect = cv2.rectangle(img, np.int32(bounding_boxes[2]), (255, 0, 0), 3)
## Green
rect = cv2.rectangle(img, np.int32(bounding_boxes[3]), (0, 255, 0), 3)

## display the segmentation mask
fig.add_subplot(rows, columns, 2)
plt.imshow(hand_mask)
plt.axis('off')
plt.title("Hand Segmentation")

# display the bounding boxes
fig.add_subplot(rows, columns, 3)
plt.imshow(rect)
plt.axis('off')
plt.title("Bounding Boxes")
plt.show()

# print(len(videos.iloc[videoNumber].loc['labelled_frames'][0]))
