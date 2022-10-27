import numpy as np

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
    print(box_xyxy)
    box_xyxy[0] = max(1, box_xyxy[0])
    box_xyxy[1] = max(1, box_xyxy[1])
    box_xyxy[2] = min(1280, box_xyxy[2])
    box_xyxy[3] = min(720, box_xyxy[3])
    box_xywh = np.array([box_xyxy[0], box_xyxy[1], box_xyxy[2]-box_xyxy[0]+1, box_xyxy[3]-box_xyxy[1]+1])
    return box_xywh


