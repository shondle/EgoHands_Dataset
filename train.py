"""
train.py allows you to train the model in model.py(in this examply, UNet), and test it off
of the validation dataset specified in utils.py.

You can save your trained model with checkpoints, and load in previously trained models
by setting LOAD_MODEL to either True (use the checkpoint) or False(retrain the model).
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from fast_scnn_model import FastSCNN
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    # check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKERS = 2
# IMAGE_HEIGHT = 90
# IMAGE_WIDTH = 160
IMAGE_HEIGHT = 161
IMAGE_WIDTH = 161
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """train_fn trains the model in model.py with the specified loader, model
    optimizer, loss function, and scaler value
    """

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets[:, :, :, :, 0]/255

        print(f"Shape of the data is {data.shape}")
        print(f"Shape of the targets is {targets.shape}")

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            print(f"Shape of the predictions is {predictions.shape}")
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    """
    main executes the program, starting off by defining transformations needed for the datasets,
    and calling functions to train and test the model
    """

    train_transform = A.Compose(
        [
            A.Resize(height=161, width=161), #IMAGE_HEIGHT - 720 ; IMAGE_WIDTH - 1280
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=161, width=161),  #IMAGE_HEIGHT - 720 ; IMAGE_WIDTH - 1280
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = FastSCNN(in_channels=3, num_classes=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        BATCH_SIZE,
        train_transform,
        val_transforms,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):

        if LOAD_MODEL is not True:
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
