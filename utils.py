"""
utils.py contains functions called by train.py in order to train and test the model in
model.py
"""

import torch
import torchvision
from dataset import EgoHandsDataset
from get_meta_by import get_meta_by
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """save_checkpoint saves a checkpoint for a trained model"""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """load_checkpoint allows you to load a previously trained model"""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    batch_size,
    train_transform,
    val_transform,
):
    """Specify your training and validation dataset for training and testing your model"""

    # training dataset
    train_ds = EgoHandsDataset(
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
        train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    # validation dataset
    val_ds = EgoHandsDataset(
        # switched S and B
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'S', 'Partner', 'B'),
        val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    """Checks the accuracy between the ground truth and predicted binary segmentation masks."""
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds = preds[0]
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """saves the predictions of both the ground state and predicted binary segmentation masks
    from the images in the validation dataset
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        print(x.shape)
        with torch.no_grad():
            preds = model(x)
            preds = preds[0]
            preds = torch.sigmoid(preds)
            preds = (preds>0.5).float()
        y = torch.movedim(y, 3, 1)
        torchvision.utils.save_image(y.float(), f"{folder}{idx}.png")

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        print(y.shape)
    model.train()
