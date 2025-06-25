import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def make_divisible(value, divisor=8, min_value=None, min_ratio= 0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels= in_places, out_channels= places, kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion = 4, bias: bool= False, channel_ratio: float = 1.0):
        super(ResNet,self).__init__()
        self.bias = bias
        self.num_classes = num_classes
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= make_divisible(64 * channel_ratio))

        # self.layer1 = self.make_layer(in_places = 64,  places= 64, block=blocks[0], stride=1)
        # self.layer2 = self.make_layer(in_places = 256, places=128, block=blocks[1], stride=2)
        # self.layer3 = self.make_layer(in_places = 512, places=256, block=blocks[2], stride=2)
        # self.layer4 = self.make_layer(in_places = 1024,places=512, block=blocks[3], stride=2)

        self.layer1 = self.make_layer(in_places = make_divisible(64 * channel_ratio),  places= make_divisible(64 * channel_ratio), block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = make_divisible(256 * channel_ratio), places= make_divisible(128 * channel_ratio), block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places = make_divisible(512 * channel_ratio), places= make_divisible(256 * channel_ratio), block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places = make_divisible(1024 * channel_ratio),places=make_divisible(512 * channel_ratio), block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def MLP(self, input):
        self.weight = nn.Parameter(torch.empty(size= (input.shape[1], self.num_classes))).to(device= input.device)
        nn.init.xavier_uniform_(self.weight.data, gain= 1.414)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 输出结果
        self.MLP(x)
        out = torch.matmul(x, self.weight)
        return out

def ResNet50(num_classes= 2, channel_ratio: float = 1.0):
    return ResNet(blocks= [3, 4, 6, 3], num_classes= num_classes, channel_ratio= channel_ratio)

def ResNet101(num_classes, channel_ratio: float = 1.0):
    return ResNet(blocks= [3, 4, 23, 3], num_classes= num_classes, channel_ratio= channel_ratio)

def ResNet152(num_classes, channel_ratio: float = 1.0):
    return ResNet(blocks= [3, 8, 36, 3], num_classes= num_classes, channel_ratio= channel_ratio)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ResNet101(num_classes= 8, channel_ratio= 1).to(device= device)

    # input = torch.randn(1, 3, 512, 1024).to(device= device)
    input = torch.rand(1, 3, 512, 512).to(device= device)
    out = model(input)
    print(out.shape)