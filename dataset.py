import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from getTrainingImgs import getTrainingMask, getTrainingImage
from getMetaBy import getMetaBy
from matplotlib import pyplot as plt
import cv2



class EgoHandsDataset(Dataset): ## inherits from Dataset

    ## specify which environments you want images and frames to train from
    def __getImages__(self, videoNum, frameNum, videos):
        videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
        masks = []
        imgs = []
        count = 0
        for i in range(len(videos)):
            for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
                masks[count] = getTrainingMask(i, j, i)
                imgs[count] = getTrainingImage(i, j, i)
                count += 1
        for i in range(len(masks)):
            fig = plt.figure(figsize=(4, 7))
            fig.add_subplot(1, 1, 1)
            plt.imshow(masks[i])
            plt.axis('off')
            plt.title("Hand Segmentation")
            plt.show()



    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) ##grayscale for PILL
        # 0.0, 255.0
        mask[mask == 255.0] = 1.0 ## for sigmoid, makes correct for labels

        if self.transofrm is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            return image, mask

def __getImages__():
    videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
    masks = []
    imgs = []
    # for i in range(len(videos)):
    #     for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
    #         masks.append(getTrainingMask(i, j, videos))
    #         imgs.append(getTrainingImage(i, j, videos))
    #         count += 1
    for i in range(1):
        for j in range(10):
            masks.append(getTrainingMask(i, j, videos))
            imgs.append(getTrainingImage(i, j, videos))
            fig = plt.figure(figsize=(4, 7))
            fig.add_subplot(1, 1, 1)
            plt.imshow(imgs[j])
            plt.axis('off')
            plt.title("Hand Segmentation")
            plt.show()

__getImages__()
