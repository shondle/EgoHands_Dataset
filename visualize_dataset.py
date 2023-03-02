"""
visualize_dataset.py gives an example of how to create and visualize a PyTorch dataset object
with respective colored images and hand segmentation labels for the
EgoHands dataset videos.
"""

import math
from matplotlib import pyplot as plt
from dataset import EgoHandsDataset
from get_meta_by import get_meta_by


# this is to query the videos. For more info, check out get_meta_by.py
videos = get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S')

# create the dataset object
dataset = EgoHandsDataset(videos, None)


# create a figure to display all the images and segmentation masks
# from the dataset you just created. This displays all frames' images and masks
# from all videos queried, so it is more of just a way to debug. Don't rely on
# this for large databases, or adjust the code if you do decide to do so.

figure = plt.figure(figsize=(10,10))
# TOTAL_IMGS = len(dataset.masks) + len(dataset.images)
TOTAL_IMGS = len(dataset.images)
# cols, rows = math.ceil(math.sqrt(TOTAL_IMGS)), math.ceil(math.sqrt(TOTAL_IMGS))
cols, rows = 8, 4
COUNT = 49
i = 1

while i < len(range(1, cols * rows + 1)):
    #img, mask = dataset.__getitem__(COUNT)
    img, mask = dataset.__getitem__(COUNT)

    figure.add_subplot(rows, cols, i)
    plt.imshow(img)
    plt.axis("off")

    # figure.add_subplot(rows, cols, i)
    # plt.imshow(mask)
    # plt.axis("off")

    # figure.add_subplot(rows, cols, i + 1)
    # plt.imshow(img)
    # plt.axis("off")
    # i += 2
    i+=1
    if COUNT < (len(dataset.images) - 1):
        COUNT = COUNT + 1

plt.show()
