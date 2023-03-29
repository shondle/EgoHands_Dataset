"""
these functions are called from dataset.py
to create the PyTorch database object.

get_training_image returns the colored image
from video and frame number specified
"""

import cv2
from EgoHands_Dataset.get_frame_path import get_frame_path
from EgoHands_Dataset.get_segmentation_mask import get_segmentation_mask


def get_training_image(video_num, frame_num, videos):
    """Retrieves the raw images for each frame in the videos"""
    video_num = video_num - 1
    frame_num = frame_num - 1
    img = cv2.imread(str(get_frame_path(videos.iloc[video_num], frame_num)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_training_mask(video_num, frame_num, videos):
    """Get the binary segmentation masks for each frame from videos queried"""
    video_num = video_num - 1
    frame_num = frame_num - 1

    hand_mask = get_segmentation_mask(videos.iloc[video_num], frame_num, 'all')

    return hand_mask
