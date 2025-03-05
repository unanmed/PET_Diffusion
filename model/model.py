import torch
import torch.nn as nn
from attention import SEBlock, AttentionBlock

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class ConvBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)

class PETUNet(nn.Module):
    def __init__(self, features=[32, 64, 128, 256], in_channels=1, out_channels=1):
        super(PETUNet, self).__init__()
        
        # 下采样部分 (in_channels 为 1)
        self.downs = nn.ModuleList()
        
        channels_num = in_channels
        for feature in features:
            self.downs.append(ConvBlockAttention(channels_num, feature))
            channels_num = feature
        
        # 上采样部分
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        prev_feature = features[-1]
        for feature in reversed(features[0:-1]):
            self.ups.append(
                UpBlock(in_channels=prev_feature, out_channels=feature)
            )
            self.attentions.append(
                SEBlock(in_channels=feature)
            )
            self.up_convs.append(
                ConvBlock(in_channels=prev_feature, out_channels=feature)
            )
            prev_feature = feature
        
        # 最终输出单通道 (out_channels = 1)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skips = []
        
        # 下采样
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # 上采样
        prev_d = skips.pop()
        for up, attention, up_conv in zip(self.ups, self.attentions, self.up_convs):
            now_x = skips.pop()
            d = up(prev_d)
            d = torch.cat((now_x, d), dim=1)  # 先拼接特征
            d = up_conv(d)  # 再卷积
            d = attention(d)  # 最后通过 SE Block
            prev_d = d
        
        # 最后输出 (保持单通道)
        x = self.final_conv(prev_d)
        return x