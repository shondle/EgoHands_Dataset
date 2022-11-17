import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from model import Net
import albumentations as A
import math
from matplotlib import pyplot as plt
from dataset import EgoHandsDataset
from get_meta_by import get_meta_by

# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_images
# )

batch_size=4
IMAGE_HEIGHT = 90
NUM_WORKERS = 2
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = False
"""what do I do here when it asks for directories? 32:11"""

def train_fn(trainloader, net, optimizer):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):  # loop over the dataset multiple times to train the network

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            print(inputs.shape)
            print(labels.shape)

            inputs = inputs.float().to()
            labels = labels.float().to()


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            target = torch.argmax(outputs, dim=1)
            print(outputs.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print('Finished Training')

        PATH = './ego_net.pth'
        torch.save(net.state_dict(), PATH)

def main():
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_ds = EgoHandsDataset(
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
        transform
    )

    print('hi')
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    val_ds = EgoHandsDataset(
        get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
        transform
    )

    testloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    for epoch in range(2):
        train_fn(train_ds, net, optimizer)
        net.load_state_dict(torch.load('./ego_net.pth'))
        check_accuracy(testloader, net)
        save_predictions_as_imgs(testloader, net, folder="saved_images/")


    # val_transforms = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

## better metrics for measurements at 44:16 because black
## background is a large part of the screen



def check_accuracy(loader, model,device="cpu"):
    correct = 0
    total = 0
    net = model
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for data in loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()
            num_pixels += torch.numel(preds)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f"Got {correct}/{num_pixels} with acc {correct/num_pixels*100:.2f}"
    )
    model.train()

    ## 46:28 visualization of what the model is actually doing
    ## bring back utils commented out code?


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cpu"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx} ")
    model.train()


if __name__ == "__main__":
    main()

