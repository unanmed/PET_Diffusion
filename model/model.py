import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class FusionConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU()
        )
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.LayerNorm(dim),
            nn.ELU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, need_weights=False)[0]
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0, trans=False):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch // 2)
        self.dilated = DilatedConvBlock(in_ch, out_ch // 2)
        self.fusion = FusionConvBlock(out_ch, out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.pool = nn.MaxPool2d(2)
        self.transform = TransformerBlock(out_ch, num_heads=8) if trans else None
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dilated(x)
        x = self.fusion(x1, x2)
        if self.drop:
            x = self.drop(x)
        x = self.pool(x)
        if self.transform:
            x = self.transform(x)
        return x
    
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class UpSampleBlock(nn.Module):  
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.ELU()
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.out_ch = out_ch  # 添加out_ch属性
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        return self.conv(x)
    
class BottleneckTransformer(nn.Module):
    def __init__(self, in_ch, hidden_dim=1024):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1)
        )
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, 
                nhead=8, 
                dim_feedforward=hidden_dim*2,
                batch_first=True
            ),
            num_layers=6
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_ch, 1),
            nn.InstanceNorm2d(in_ch),
            nn.ELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.input(x)
        x = x.view(B, H * W, C)
        x = self.trans(x)
        x = x.view(B, C, H, W)
        x = self.output(x)
        return x

class PETUNet(nn.Module):
    def __init__(self, base_ch=64, in_ch=1, out_ch=1):
        super(PETUNet, self).__init__()
        self.in_conv = ConvBlock(in_ch, base_ch)
        self.down1 = Encoder(base_ch, base_ch * 2, dropout=0.2)
        self.down2 = Encoder(base_ch * 2, base_ch * 4, dropout=0.2)
        self.down3 = Encoder(base_ch * 4, base_ch * 8, dropout=0.2)
        self.down4 = Encoder(base_ch * 8, base_ch * 16, dropout=0.2)

        self.bottleneck = BottleneckTransformer(base_ch * 16, hidden_dim=1024)
        self.fusion = ConvBlock(base_ch * 32, base_ch * 16)

        # 上采样层
        self.up1 = Decoder(base_ch * 16, base_ch * 8)
        self.up2 = Decoder(base_ch * 8, base_ch * 4)
        self.up3 = Decoder(base_ch * 4, base_ch * 2)
        self.up4 = Decoder(base_ch * 2, base_ch)
        self.out_conv = OutConv(base_ch, out_ch)

    def forward(self, x):
        # 编码器路径
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 瓶颈层
        x5_cnn = self.down4(x4)
        x5_trans = self.bottleneck(x5_cnn)
        x5 = self.fusion(torch.cat([x5_cnn, x5_trans], dim=1))

        # 解码器路径并调整通道
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.out_conv(x)
        return F.tanh(x)

# 检查显存占用
if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256)).cuda()
    model = PETUNet(in_ch=1, out_ch=1).cuda()
    print_memory("初始化后")
    output = model(x)
    print_memory("前向传播后")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")