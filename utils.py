"""
utils.py contains functions called by train.py in order to train and test the model in
model.py
"""

import torch
import torchvision
from torchvision.utils import draw_segmentation_masks
from dataset import EgoHandsDataset
from get_meta_by import get_meta_by
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image

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
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
        val_transform
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    clean_val_ds = EgoHandsDataset(
        # switched S and B
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S')
    )

    clean_val_loader = DataLoader(
        clean_val_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader, clean_val_loader

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
            preds = torch.sigmoid(model(x))
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
    clean_loader, loader, model, folder="saved_images/", device="cuda"
):
    """saves the predictions of both the ground state and predicted binary segmentation masks
    from the images in the validation dataset
    """
    model.eval()

    for idx, (loader_item, clean_loader_item) in enumerate(zip(loader, clean_loader)):
        x1, y1 = clean_loader_item
        x, y = loader_item
        x = x.to(device=device)
        print(x.shape)
        with torch.no_grad():
#             print(f"This is the positive version: {torch.positive(model(x))}")
            print(f"This is before the sigmoid: {torch.max(model(x))}")
            preds = torch.sigmoid(model(x))
            print(f"This is after the sigmoid: {torch.max(preds)}")
            preds = (preds>0.5).float()
        y = torch.movedim(y, 3, 1)

        torchvision.utils.save_image(y.float(), f"{folder}{idx}.png")

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        preds = preds.cpu()

        class_dim = 0
        print(f"Shape of preds is {preds.shape}")
        preds = F.interpolate(preds, size=(720, 1280), mode = 'nearest')
        # print(f"shape of y is {y.shape}")
        # y = F.interpolate(y, size=(720, 1280))
        # print(f"shape of y is {y.shape}")

        bool_hand_masks = (preds[:, 0,  :, :] == 1)
        # bool_ground_masks = (y[:, 0,  :, :] == 1)

        # plt.imshow(x[0, :, :, :].permute(1, 2, 0))
        print(bool_hand_masks.shape)
        # plt.imshow(bool_hand_masks[0 , :, :])

        for img, mask in zip(x1, bool_hand_masks):
            # pil_img = Image.fromarray(img)
            print(img)
            print(img.shape)
            print(f"The shape of the y bool ground is {y.shape}.shape and x1 is {x1.shape} and bool hand mask is {bool_hand_masks.shape}")
            print(f"Unique is {torch.unique(y)}")
        hands_with_masks = [
            draw_segmentation_masks(img.permute(2, 0, 1), masks=mask, alpha=0.5, colors="yellow")
            for img, mask in zip(x1, bool_hand_masks)
        ]

        images = [
            img.permute(2, 0, 1)
            for img, mask in zip(x1, bool_hand_masks)
        ]

        masks = [
            mask
            for img, mask in zip(x1, y)
        ]

        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[1].imshow(hands_with_masks[0].permute(1, 2, 0))
        axs[1].set_title('Predicted Segmentation Mask')
        axs[1].axis('off')

        axs[0].imshow(images[0].permute(1, 2, 0))
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[2].imshow(masks[0].permute(1, 2, 0))
        axs[2].set_title('Ground Truth Mask')
        axs[2].axis('off')

        # plt.imshow(preds[0,:, :, :].permute(1, 2, 0), alpha = 0.6)
        plt.savefig("img3.jpg", dpi=300)
        plt.show()

        print(y.shape)
    model.train()
