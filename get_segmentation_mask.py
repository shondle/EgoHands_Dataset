"""
get_segmentation_mask(vid, i, hand_type) returns the binary segmentation mask
for hands in the "i"th frame in video "vid", where "vid" is an EgoHands video
metadata structure.

img_mask = get_segmentation_mask(vid, 1, 'all') returns a mask for all hands in first frame of vid

img_mask = get_segmentation_mask(vid, 1, 'mine') returns a
mask for all egocentric observer hands in first frame of vid

img_mask = get_segmentation_mask(vid, 1, 'your_right') returns a mask for
all egocentric partner's right hand in first frame of vid

Possible values for hand_type are
"all", "mine", "yours", "my_left", "my_right", "your_left", "your_right".


For full dataset details, see the
<a href="matlab: web('http://vision.soic.indiana.edu/egohands')">EgoHands project website</a>.

See also get_frame_path, get_meta_by, get_bounding_boxes, showLabelsOnFrame
"""

import numpy as np
import cv2


def get_segmentation_mask(video, i, hand_type):
    """Retrieves the binary segmentation mask for your query"""
    img_mask = np.zeros([720, 1280, 3], dtype= "uint8")
    if (hand_type == 'my_left' or hand_type=='mine' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][1])):
        shape = np.int32(video.loc['labelled_frames'][0][i][1])
        # all make a white mask
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'my_right' or hand_type=='mine' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][2])):
        shape = np.int32(video.loc['labelled_frames'][0][i][2])
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'your_left' or hand_type == 'yours' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][3])):
        shape = np.int32(video.loc['labelled_frames'][0][i][3])
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'your_right' or hand_type == 'yours' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][4])):
        shape = np.int32(video.loc['labelled_frames'][0][i][4])
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))

    return img_mask
