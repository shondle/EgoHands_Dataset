from getMetaBy import getMetaBy
from getFramePath import getFramePath
from getSegmentationMask import getSegmentationMask
from getBoundingBoxes import getBoundingBoxes
import numpy as np
import cv2
from matplotlib import pyplot as plt

videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')

fig = plt.figure(figsize=(4, 7))
rows = 3
columns = 1


img = cv2.imread(str(getFramePath(videos.iloc[0], 7)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Video: " + (videos.iloc[0]).loc['video_id'][0] + " - Frame #" + "8")

hand_mask = getSegmentationMask(videos.iloc[0], 7, 'all')
bounding_boxes = getBoundingBoxes(videos.iloc[0], 7)

## Blue
rect = cv2.rectangle(img, np.int32(bounding_boxes[0]), (0, 0, 255), 3)
## Yellow
rect = cv2.rectangle(img, np.int32(bounding_boxes[1]), (255, 255, 0), 3)
## Red
rect = cv2.rectangle(img, np.int32(bounding_boxes[2]), (255, 0, 0), 3)
## Green
rect = cv2.rectangle(img, np.int32(bounding_boxes[3]), (0, 255, 0), 3)

fig.add_subplot(rows, columns, 2)
plt.imshow(hand_mask)
plt.axis('off')
plt.title("Hand Segmentation")

fig.add_subplot(rows, columns, 3)
plt.imshow(rect)
plt.axis('off')
plt.title("Bounding Boxes")
plt.show()

