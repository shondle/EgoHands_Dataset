import scipy.io as sio
import numpy as np
import cv2


def getSegmentationMask(video, i, hand_type):
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


