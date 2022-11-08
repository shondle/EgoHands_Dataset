import numpy as np

 # getBoundingBoxes(vid, i) returns the ground-truth bounding boxes for the "ith" annotated frame of
 # video "vid", where "vid" is an EgoHands video metadata structure.
 #
 # Boxes is a 4x4 matrix, where each row corresponds to a hand bounding box in the format [x y width
 # height], where x and y mark the top left corner of the box. The rows from top to bottom contain
 # the bounding boxes for "own left", "own right", "other left", and "other right" hand respectively.
 # If a hand is not in the frame, the values are set to 0.
 #
 #
 #   For full dataset details, see the <a href="matlab: web('http://vision.soic.indiana.edu/egohands')">EgoHands project website</a>.
 #
 #   See also getFramePath, getMetaBy, getSegmentationMask, showLabelsOnFrame


def getBoundingBoxes(video, i):
    boxes = np.zeros([4, 4])
    if np.any(video.loc['labelled_frames'][0][i][1]):
        box = segmentation2box(np.int32(video.loc['labelled_frames'][0][i][1]))
        boxes[0, :] = box
    if np.any(video.loc['labelled_frames'][0][i][2]):
        box = segmentation2box(np.int32(video.loc['labelled_frames'][0][i][2]))
        boxes[1, :] = box
    if np.any(video.loc['labelled_frames'][0][i][3]):
        box = segmentation2box(np.int32(video.loc['labelled_frames'][0][i][3]))
        boxes[2, :] = box
    if np.any(video.loc['labelled_frames'][0][i][4]):
        box = segmentation2box(np.int32(video.loc['labelled_frames'][0][i][4]))
        boxes[3, :] = box
    return boxes

def segmentation2box(shape):
    box_xyxy = np.round(np.array([np.min(shape[:, 0]), np.min(shape[:, 1]), np.max(shape[:, 0]), np.max(shape[:, 1])]))
    box_xyxy[0] = max(1, box_xyxy[0])
    box_xyxy[1] = max(1, box_xyxy[1])
    box_xyxy[2] = min(1280, box_xyxy[2])
    box_xyxy[3] = min(720, box_xyxy[3])
    box_xywh = np.array([box_xyxy[0], box_xyxy[1], box_xyxy[2]-box_xyxy[0]+1, box_xyxy[3]-box_xyxy[1]+1])
    return box_xywh


