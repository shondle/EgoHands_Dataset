import scipy.io as sio
import numpy as np
import cv2


def getSegmentationMask(video, i, hand_type):
    img_mask = np.zeros([720, 1280, 3], dtype= "uint8")
    if (hand_type == 'my_left' or hand_type=='mine' or hand_type == 'all'
            and not np.any(video.loc['labelled_frames'][0][i][1])):
        shape = reshapeAreaCoords(video.loc['labelled_frames'][0][i][1])
        # all make a white max
        img_mask = cv2.fillpoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'my_right' or hand_type=='mine' or hand_type == 'all'
            and not np.any(video.loc['labelled_frames'][0][i][2])):
        shape = video.loc['labelled_frames'][0][i][2]
        img_mask = cv2.fillpoly(img_mask, pts = [shape], color=(255, 255, 255))
    if (hand_type == 'your_left' or hand_type == 'yours' or hand_type == 'all'
            and not np.any(video.loc['labelled_frames'][0][i][3])):
        shape = video.loc['labelled_frames'][0][i][3]
        img_mask = cv2.fillpoly(img_mask, pts=[shape], color=(255, 255, 255))
    if (hand_type == 'your_right' or hand_type == 'yours' or hand_type == 'all'
            and not np.any(video.loc['labelled_frames'][0][i][4])):
        shape = video.loc['labelled_frames'][0][i][4]
        img_mask = cv2.fillpoly(img_mask, pts=[shape], color=(255, 255, 255))

    # how to do logical() here?
    img_mask = img_mask[:, :, 0]

    return img_mask


def reshapeAreaCoords(shape):
    shape2 = np.zeros([1][2*len(shape)])
    ## ' for conjugate transpose
    # https://stackoverflow.com/questions/51106981/conjugate-transpose-of-self-using-numpy-syntax
    shape2[0:1:-1] = shape[:, 0].conj().T
    shape2[1:1:-1] = shape[:, 1].conj().T


