import torch
import torchvision
from dataset import EgoHandsDataset
from get_meta_by import get_meta_by
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    # train_dir,
    # train_maskdir,
    # val_dir,
    # val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    # num_workers=4,
    # pin_memory=True,
):
    train_ds = EgoHandsDataset(
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
        train_transform
    )

    # train_loader = DataLoader(
    #     train_ds,
    #        batch_size=batch_size,
    #     num_workers=4,
    #     pin_memory=True,
    #     shuffle=True,
    # )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    val_ds = EgoHandsDataset(
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
        val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
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
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        print(x.shape)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()
        y = torch.movedim(y, 3, 1)
        y = torch.movedim(y, 2, 3)
        torchvision.utils.save_image(preds, f"{folder}/pred{idx}.png")
        torchvision.utils.save_image(y.float(), f"{folder}{idx}.png")

        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )

        print(y.shape)
    model.train()
        # _, h, w, _ = y.shape  # assumes y's torch shape is (h,w) and NOT (batch_size, h, w)
        # y_img = torch.zeros(3, h, w)
        # y_img[0, :, :] = y
        # y_img[1, :, :] = y
        # y_img[2, :, :] = y

        # _, h, w, _ = y.shape  # assumes y's torch shape is (h,w) and NOT (batch_size, h, w)
        # y_img = torch.zeros(16, h, w, 3)
        # y_img[:, :, :, 1] = y[:, :, :, 0]
        # y_img[:, :, :, 2] = y[:, :, :, 0]
        # y_img[:, :, :, 3] = y[:, :, :, 0]
