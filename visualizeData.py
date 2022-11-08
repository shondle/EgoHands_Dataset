from dataset import EgoHandsDataset
from getMetaBy import getMetaBy
from matplotlib import pyplot as plt
import math

# This is an example of how to create and visualize a PyTorch dataset object
# with respective colored images and hand segmentation labels for the
# EgoHands dataset videos.

# this is to query the videos. For more info, check out getMetaBy.py
videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')

# create the dataset object
dataset = EgoHandsDataset(videos, None)


# create a figure to display all the images and segmentation masks
# from the dataset you just created. This displays all frames' images and masks
# from all videos queried, so it is more of just a way to debug. Don't rely on
# this for large databases, or adjust the code if you do decide to do so.

figure = plt.figure(figsize=(10,10))
totalImgs = len(dataset.masks) + len(dataset.images)
cols, rows = math.ceil(math.sqrt(totalImgs)), math.ceil(math.sqrt(totalImgs))
count = 1
i = 1

while i < len(range(1, cols * rows + 1)):
    img, mask = dataset.__getitem__(count)

    figure.add_subplot(rows, cols, i)
    plt.imshow(mask)
    plt.axis("off")

    figure.add_subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis("off")
    i += 2
    if count < (len(dataset.images) - 1):
        count = count + 1

plt.show()
