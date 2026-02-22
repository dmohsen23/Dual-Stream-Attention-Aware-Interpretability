from __future__ import print_function, division
import torch.nn as nn
import torchvision.models as models

class Resnet50(nn.Module):
    def __init__(self, num_cls=3, in_channels=3, in_size=(224, 224), out_channels=2048, use_rgb=True, weights=None):
        super(Resnet50, self).__init__()
        self.num_cls = num_cls
        # get the pretrained Resnet50 network
        self.net = models.resnet50(weights=weights)
        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.net.fc.in_features
        self.out_channels = num_ftrs
        self.net.fc = nn.Linear(num_ftrs, num_cls)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        layer1 = self.net.layer1(x)
        layer2 = self.net.layer2(layer1)
        layer3 = self.net.layer3(layer2)
        layer4 = self.net.layer4(layer3)

        pooled_layer4 = self.avgpool(layer4).view(layer4.size(0), -1)

        out = self.net.fc(pooled_layer4)

        return out, layer4, pooled_layer4


