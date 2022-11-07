import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from getTrainingImgs import getTrainingMask, getTrainingImage
from getMetaBy import getMetaBy
from matplotlib import pyplot as plt
import cv2



class EgoHandsDataset(Dataset): ## inherits from Dataset

    # def __init__(self, image_dir, mask_dir, transform=None):
    #     self.image_dir = image_dir
    #     self.mask_dir = mask_dir
    #     self.transform = transform
    #     self.images = os.listdir(image_dir)

    def __init__(self, videos):
        # videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
        self.videos = videos
        masks = []
        imgs = []
        for i in range(len(videos)):
            for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
                masks.append(getTrainingMask(i, j, self.videos))
                imgs.append(getTrainingImage(i, j, self.videos))
        self.images = imgs
        self.masks = masks
    ## specify which environments you want images and frames to train from

    # def __getImages__(self):
    #     ## previous inputs: self, videoNum, frameNum, videos
    #     videos = self.videos
    #     masks = []
    #     imgs = []
    #     for i in range(len(videos)):
    #         for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
    #             masks.append(getTrainingMask(i, j, videos))
    #             imgs.append(getTrainingImage(i, j, videos))
    #     return imgs, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # img_path = os.path.join(self.image_dir, self.images[index])
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        # image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) ##grayscale for PILL
        # 0.0, 255.0
        img = self.images[index]
        mask = self.masks[index]
        mask[mask == 255.0] = 1.0 ## for sigmoid, makes correct for labels

        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

# def __getImages__():
#     videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
#     masks = []
#     imgs = []
#     for i in range(len(videos)):
#         for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
#             masks.append(getTrainingMask(i, j, videos))
#             imgs.append(getTrainingImage(i, j, videos))
#             fig = plt.figure(figsize=(4, 7))
#             fig.add_subplot(1, 1, 1)
#             plt.imshow(imgs[j])
#             plt.axis('off')
#             plt.title("Hand Segmentation")
#             plt.show()
    # for i in range(1):
    #     for j in range(10):
    #         masks.append(getTrainingMask(i, j, videos))
    #         imgs.append(getTrainingImage(i, j, videos))
    #         fig = plt.figure(figsize=(4, 7))
    #         fig.add_subplot(1, 1, 1)
    #         plt.imshow(imgs[j])
    #         plt.axis('off')
    #         plt.title("Hand Segmentation")
    #         plt.show()
