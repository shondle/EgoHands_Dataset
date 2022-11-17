"""
This is the code which created the PyTorch dataset object with
colored image and respective segmentation labels. Demonstration of how to create and
object and visualize the data from visualize_dataset.py
"""

from torch.utils.data import Dataset
from get_training_imgs import get_training_mask, get_training_image


class EgoHandsDataset(Dataset):
    """Create PyTorch dataset object of queried videos"""

    def __init__(self, videos, transform=None):
        self.videos = videos
        masks = []
        imgs = []
        for i in range(len(videos)):
            for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
                masks.append(get_training_mask(i + 1, j + 1, self.videos))
                imgs.append(get_training_image(i + 1, j + 1, self.videos))
        # you can see the masks here
        self.images = imgs
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]

        if self.transform is not None:
            augmentations = self.transform(imgage=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        return img, mask
