from dataset import EgoHandsDataset
from getMetaBy import getMetaBy
from matplotlib import pyplot as plt
import math

videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')

dataset = EgoHandsDataset(videos, None)

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
