import scipy.io as sio
import numpy as np
import cv2


def getSegmentationMask(video, i, hand_type):
    img_mask = np.zeros([720, 1280, 3], dtype= "uint8")
    if (hand_type == 'my_left' or hand_type=='mine' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][1])):
        shape = reshapeAreaCoords(video.loc['labelled_frames'][0][i][1])
        # all make a white mask
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'my_right' or hand_type=='mine' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][2])):
        shape = reshapeAreaCoords(video.loc['labelled_frames'][0][i][2])
        img_mask = cv2.fillPoly(img_mask, pts = [shape], color=(255, 255, 255))
    if (hand_type == 'your_left' or hand_type == 'yours' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][3])):
        shape = reshapeAreaCoords(video.loc['labelled_frames'][0][i][3])
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'your_right' or hand_type == 'yours' or hand_type == 'all'
            and np.any(video.loc['labelled_frames'][0][i][4])):
        shape = reshapeAreaCoords(video.loc['labelled_frames'][0][i][4])
        img_mask = cv2.fillPoly(img_mask, pts=[shape], color=(255, 255, 255))

    # how to do logical() here?
    img_mask = img_mask[:, :, 0].astype(bool)
    img_mask = img_mask.astype(np.uint8)

    return img_mask


def reshapeAreaCoords(shape):
    shape2 = np.zeros((1, 2*len(shape)))
    ## not fitting into shape 2, how do I fix this?
    shape2[ ::2] = shape[:, 0].conj().T
    shape2[1:-1:2] = shape[:, 1].conj().T
    return shape2


