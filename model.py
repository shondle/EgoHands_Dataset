# """
#     Maintainer: Selma Wanna  (slwanna@utexas.edu)
#     This file creates the FastSCNN model architecture.
# """
# import torch
# import torch.nn as nn
# from stochman import nnj


# class UNET(nn.Module):
#     """
#     Constructs the FastSCNN model
#     """
#     ## had to change int to 1
#     def __init__(self, last_layer: bool=False, num_classes:int=1):
#         super().__init__()

#         self.last_layer = last_layer
#         self.num_classes = num_classes

#         # Model layer sizing
#         img_h = img_w = 64
#         self.dw_channels1 = int(32/2)
#         self.dw_channels2 = int(64/2)
#         self.out_channels = int(128/2)
#         self.intermediate_channel = int(96/2)
#         t = int(6/2)

#         """
#         FastSCNN full architecture
#         """
#         self.deterministic = nnj.Sequential(
#             # 1. Learning to Downsample:
#             nnj.Conv2d(3, self.dw_channels1, 1, bias=False),
#             nnj.Tanh(),
#             nnj.Conv2d(self.dw_channels1, self.dw_channels1, 1, bias=False),
#             nnj.Tanh(),
#             nnj.Conv2d(self.dw_channels1, self.dw_channels2, 1, bias=False),
#             nnj.Tanh(),
#             nnj.Conv2d(self.dw_channels2, self.dw_channels2, 1, bias=False),
#             nnj.Tanh(),
#             nnj.Conv2d(self.dw_channels2, self.out_channels, 1, bias=False),
#             nnj.Tanh(),
#             nnj.Flatten(),
#             nnj.SkipConnection(
#                 nnj.Reshape(self.out_channels,img_h,img_h),
#                 # 2. Global Feature Extractor:
#                 #     a) linear bottle neck
#                 nnj.Conv2d(self.out_channels,self.dw_channels2*t,1, bias=False),
#                 nnj.Tanh(),
#                 nnj.Conv2d(self.dw_channels2*t,self.dw_channels2*t,1, bias=False),
#                 nnj.Tanh(),
#                 nnj.Conv2d(self.dw_channels2*t,self.dw_channels2,1,bias=False),
#                 #     b) linear bottle neck
#                 nnj.Conv2d(self.dw_channels2,self.intermediate_channel*t,1, bias=False),
#                 nnj.Tanh(),
#                 nnj.Conv2d(self.intermediate_channel*t,self.intermediate_channel*t,1, bias=False),
#                 nnj.Tanh(),
#                 nnj.Conv2d(self.intermediate_channel*t,self.intermediate_channel,1,bias=False),
#                 #     c) linear bottle neck
#                 nnj.Conv2d(self.intermediate_channel,self.out_channels*t,1, bias=False),
#                 nnj.Tanh(),
#                 nnj.Conv2d(self.out_channels*t,self.out_channels*t,1, bias=False),
#                 nnj.Tanh(),
#                 nnj.Conv2d(self.out_channels*t,self.out_channels*t,1,bias=False),
#                 #     d) pyramid pooling layer -- Can't work w/ Sequential, needs 4 features
#                 nnj.MaxPool2d(1),
#                 nnj.Conv2d(self.out_channels*t,self.out_channels,1, bias=False), 
#                 nnj.Upsample(scale_factor=2), # feature 1
#                 # nnj.MaxPool2d(2),
#                 # nnj.Conv2d(128,128,1, bias=False), 
#                 # nnj.Upsample(scale_factor=2),  # feature 2
#                 # nnj.MaxPool2d(3),
#                 # nnj.Conv2d(128,128,1, bias=False), 
#                 # nnj.Upsample(scale_factor=2), # feature 3
#                 # nnj.MaxPool2d(6),
#                 # nnj.Conv2d(768,128,1, bias=False), 
#                 # nnj.Upsample(scale_factor=2), # feature 4

#                 nnj.Flatten(),
#                 add_hooks=True,
#             ),
#             # Skipping Feature fusion
#             # 3. Feature Fusion:
#             nnj.Reshape(320,img_h,img_h),

#             # # 4. Classifier
#             nnj.Conv2d(320,img_h,1),
#             nnj.Tanh(),
#             nnj.Conv2d(img_h,img_h,1),
#             nnj.Tanh(),
#             nnj.Conv2d(img_h,img_h,1),
#             nnj.Tanh(),

#             # nnj.Conv2d(128,128,1),
#             # nnj.Tanh(),
#             # nnj.Conv2d(128,128,1),
#             # nnj.Tanh(),

#             nnj.Conv2d(64,self.num_classes,1)

#         )

#     # for line_profiler, uncomment @profile line
#     # run: $ kernprof -l -v fast_scnn.py
#     # @profile
#     def forward(self, x):
#         """
#         Executes forward pass
#         """
# #         print(f"x shape in forward is {x.shape}")
#         return self.deterministic(x)


# def count_parameters(model):
#     """
#     returns the number of trainable parameters in a model
#     """
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# if __name__ == '__main__':
#     # create fake RGB image
#     img = torch.randn(2, 3, 64, 64)

#     # create the model
#     model = FastSCNN()

#     # print model summary
#     print(model)

#     # print model trainable parameter size
#     print(count_parameters(model))

#     # print output shape
#     print(model(img).shape)
    
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()