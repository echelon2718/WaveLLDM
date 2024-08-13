import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.layer1 = self.conv_instance_norm_leakyrelu(in_channels, 64, kernel_size=4, stride=2)
        self.layer2 = self.conv_instance_norm_leakyrelu(64, 128, kernel_size=4, stride=2)
        self.layer3 = self.conv_instance_norm_leakyrelu(128, 256, kernel_size=4, stride=2)
        self.final = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def conv_instance_norm_leakyrelu(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, return_features=False):
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.final(x)
        x = self.sigmoid(x)  # Apply sigmoid activation

        if return_features:
            return x, features
        return x