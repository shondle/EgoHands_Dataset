""" This demo shows how to load and access ground-truth data for any of the videos. """

import numpy as np
import cv2
from matplotlib import pyplot as plt
from get_meta_by import get_meta_by
from get_frame_path import get_frame_path
from get_segmentation_mask import get_segmentation_mask
from get_bounding_boxes import get_bounding_boxes

# Let's load all videos at the courtyard location where the activity was puzzle solving.
# get_meta_by() returns a struct array that contains all possible meta information (including
# the ground-truth data) about the videos. Check the get_meta_by() documentation for more.

# videos = get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE')
videos = get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'S', 'Partner', 'B')

# Each video has 100 annotated frames. Let's consider the first video. One can access the 8th frame
# of the first video like this:
VIDEO_NUM = 1 # enter which video you want here
FRAME_NUM = 75 # enter frame number you want here
# originally 8, then 51, then 21, then 75

VIDEO_NUM = VIDEO_NUM - 1
FRAME_NUM = FRAME_NUM - 1

# creating figure to display
fig = plt.figure(figsize=(4, 7))
ROWS = 3
COLUMNS = 1

# getting colored image
img = cv2.imread(str(get_frame_path(videos.iloc[VIDEO_NUM], FRAME_NUM)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# showing colored image on figure
fig.add_subplot(ROWS, COLUMNS, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Video: " + (videos.iloc[VIDEO_NUM]).loc['video_id'][VIDEO_NUM]
          + " - Frame #%s" % (FRAME_NUM+1))


# Here is how to get a binary mask with hand segmentations for the current frame. The third argument
# implies that the mask will show "all" hands. To get masks for specific hands, change this argument
# to e.g. "my_right" or "yours" to get only the observer's right hand or only the other actor's
# hands respectively. Check the get_segmentation_mask() documentation for more.
hand_mask = get_segmentation_mask(videos.iloc[VIDEO_NUM], FRAME_NUM, 'all')


# The bounding boxes for each hand are also easily accessible. The function below returns a 4x4
# matrix, where each row corresponds to a hand bounding box in the format [x y width height], where
# x and y mark the top left corner of the box. The ROWS from top to bottom contain the bounding
# boxes for "own left", "own right", "other left", and "other right" hand respectively. If a hand
# is not in the frame, the values are set to 0.
bounding_boxes = get_bounding_boxes(videos.iloc[VIDEO_NUM], FRAME_NUM)

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
fig.add_subplot(ROWS, COLUMNS, 2)
plt.imshow(hand_mask)
plt.axis('off')
plt.title("Hand Segmentation")

# display the bounding boxes
fig.add_subplot(ROWS, COLUMNS, 3)
plt.imshow(rect)
plt.axis('off')
plt.title("Bounding Boxes")
plt.savefig("dummy_name.png")
plt.show()

# print(len(videos.iloc[VIDEO_NUM].loc['labelled_frames'][0]))
