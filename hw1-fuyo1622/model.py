# Model definitions used by the HW1 classification and segmentation tasks.
import torch.nn as nn
import torchvision.models as models


# Basic residual block for the from-scratch ResNet-18 implementation.
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


# ResNet-18 classifier assembled from the residual block above.
class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1:
            downsample = self.downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(
                in_channels,
                out_channels,
                downsample=downsample,
                stride=stride,
            ),
            Block(out_channels, out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


# ResNet-18 variant that exposes its last convolutional feature map.
class tSNE_resnet18(nn.Module):
    def __init__(self, num_classes):
        super(tSNE_resnet18, self).__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1:
            downsample = self.downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(
                in_channels,
                out_channels,
                downsample=downsample,
                stride=stride,
            ),
            Block(out_channels, out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        s = self.layer4(x)

        x = self.avgpool(s)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return s

    def downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


# Torchvision ResNet-50 wrapper with a task-specific classification head.
class resnet50(nn.Module):
    def __init__(self, num_classes, pre_trained):
        super(resnet50, self).__init__()
        if pre_trained:
            self.model = models.resnet50(
                weights="ResNet50_Weights.IMAGENET1K_V2"
            )
        else:
            self.model = models.resnet50()
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# VGG16-based FCN-32s baseline for seven-class semantic segmentation.
class vgg16fcn32(nn.Module):
    def __init__(self, num_classes):
        super(vgg16fcn32, self).__init__()
        self.features = models.vgg16(weights="IMAGENET1K_V1").features
        self.conv1 = nn.Conv2d(512, 4096, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4096, 4096, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4096, 7, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)
        self.upsample32x = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            64,
            32,
            16,
            bias=False,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.upsample32x(s)

        return x
