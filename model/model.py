import torch
import torch.nn as nn

class PETUNet(nn.Module):
    def __init__(self, features=[128, 256, 512, 1024], in_channels=1, out_channels=1):
        super(PETUNet, self).__init__()
        
        # 下采样部分 (in_channels 为 1)
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU()
                )
            )
            in_channels = feature
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 上采样部分
        self.ups = nn.ModuleList()
        prev_feature = features[-1]
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_feature + feature, feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(feature),
                    nn.ReLU()
                )
            )
            prev_feature = feature
        
        # 最终输出单通道 (out_channels = 1)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        
        # 下采样
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        x = self.bottleneck(x)
        
        # 上采样
        for up in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        
        # 最后输出 (保持单通道)
        x = self.final_conv(x)
        return x