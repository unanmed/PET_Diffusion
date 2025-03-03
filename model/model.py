import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # F_int = max(1, F_int // 4)  # 减少中间通道数
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, g, x):
        psi = torch.sigmoid(self.psi(self.W_g(g) + self.W_x(x)))
        return x * psi
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
    def __init__(self, features=[16, 32, 64, 128], in_channels=1, out_channels=1):
        super(PETUNet, self).__init__()
        
        # 下采样部分 (in_channels 为 1)
        self.downs = nn.ModuleList()
        
        channels_num = in_channels
        for feature in features:
            self.downs.append(ConvBlock(channels_num, feature))
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
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            self.up_convs.append(
                ConvBlock(in_channels=prev_feature, out_channels=feature)
            )
            prev_feature = feature
        
        # 最终输出单通道 (out_channels = 1)
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1, stride=1)

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
            now_x = attention(g=d, x=now_x)
            d = torch.cat((now_x, d), dim=1)
            d = up_conv(d)
            prev_d = d
        
        # 最后输出 (保持单通道)
        x = self.final_conv(x)
        return x