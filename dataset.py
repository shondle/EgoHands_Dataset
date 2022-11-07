from torch.utils.data import Dataset
from getTrainingImgs import getTrainingMask, getTrainingImage

class EgoHandsDataset(Dataset):

    def __init__(self, videos, transform=None):
        self.videos = videos
        masks = []
        imgs = []
        for i in range(len(videos)):
            for j in range(len(videos.iloc[i].loc['labelled_frames'][0])):
                masks.append(getTrainingMask(i + 1, j + 1, self.videos))
                imgs.append(getTrainingImage(i + 1, j + 1, self.videos))
        ##you can see the masks here
        self.images = imgs
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]

        if self.transform is not None:
            augmentations = self.transform(img=img, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return img, mask
