import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class FusionConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
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
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.out_ch = out_ch  # 添加out_ch属性
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
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
    def __init__(self, in_ch, input_size, hidden_dim=1024):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1),
            Rearrange("b c h w -> b (h w) c")
        )
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
            hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim,  # 原为2倍，改为1倍
            batch_first=True  # 使用batch_first减少permute
         ),
            num_layers=3  # 减少层数
        )
        self.output = nn.Sequential(
            Rearrange("b (h w) c -> b c h w",  h=input_size[0], w=input_size[1]),  # 使用保存的尺寸
            nn.Conv2d(hidden_dim, in_ch, 1)
        )

    def forward(self, x):
        
        x = self.input(x)
        
        x = x.permute(1, 0, 2)
        x = self.trans(x)
        x = x.permute(1, 0, 2)
        x = self.output(x)

        return x

class PETUNet(nn.Module):
    def __init__(self, base_ch=64, in_ch=1, out_ch=1, input_size=(256, 256)):
        super(PETUNet, self).__init__()
        self.in_conv = ConvBlock(in_ch, base_ch)
        self.down1 = Encoder(base_ch, base_ch * 2, dropout=0.2)
        self.down2 = Encoder(base_ch * 2, base_ch * 4, dropout=0.2)
        self.down3 = Encoder(base_ch * 4, base_ch * 8, dropout=0.2, trans=True)

        H, W = input_size
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1),
            BottleneckTransformer(base_ch * 8, input_size=(H//8, W//8), hidden_dim=1024),
            nn.Conv2d(base_ch * 8, base_ch * 16, 1)
        )

        # 上采样层
        self.up1 = Decoder(base_ch * 16, base_ch * 8)
        self.up2 = Decoder(base_ch * 8, base_ch * 4)
        self.up3 = Decoder(base_ch * 4, base_ch * 2)
        self.up4 = Decoder(base_ch * 2, base_ch)
        self.out_conv = OutConv(base_ch, out_ch)

        # 添加通道调整层
        self.adjust_x3 = nn.Conv2d(base_ch*4, base_ch*8, 1)
        self.adjust_x2 = nn.Conv2d(base_ch*2, base_ch*4, 1)
        self.adjust_x1 = nn.Conv2d(base_ch, base_ch*2, 1)

    def forward(self, x):
        # 编码器路径
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 瓶颈层
        x5 = self.bottleneck(x4)

        # 解码器路径并调整通道
        x3 = self.adjust_x3(x3)
        x = self.up1(x5, x3)
        
        x2 = self.adjust_x2(x2)
        x = self.up2(x, x2)
        
        x1 = self.adjust_x1(x1)
        x = self.up3(x, x1)
        
        x = self.up4(x, None)
        return self.out_conv(x)
    
class BiPathResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 3, padding=1),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch//2, 3, padding=1),
        )
        self.global_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch//2, 3, padding=dilation, dilation=dilation),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x_local = self.local_path(x)
        x_global = self.global_path(x)
        x_fused = torch.cat([x_local, x_global], dim=1)
        return self.fusion(x_fused) + residual
    
class ENLSA(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_flat = x_in.view(B, -1, C)
        
        # 注意力计算
        q = self.proj_q(x_flat).reshape(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(x_flat).reshape(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(x_flat).reshape(B, -1, self.num_heads, self.head_dim)
        
        attn = torch.einsum('bnhd,bmhd->bnmh', q, k) / (self.head_dim**0.5)
        attn = attn.softmax(dim=2)
        x = torch.einsum('bnmh,bmhd->bnhd', attn, v)
        x = x.reshape(B, -1, C)
        x = self.proj_o(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # 恢复形状
        
        # MLP部分
        x_mlp = x_in + x.view(B, H, W, C)
        x_mlp = self.norm(x_mlp)
        x_mlp = self.mlp(x_mlp).permute(0, 3, 1, 2)
        
        return x + x_mlp
    
class SpatialCrossScaleIntegrator(nn.Module):
    def __init__(self, channels, embed_dim=256):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, 1) for ch in channels
        ])
        self.inv_proj_layers = nn.ModuleList([
            nn.Conv2d(embed_dim, ch, 1) for ch in channels
        ])
        self.transformer = TransformerBlock(embed_dim * len(channels), num_heads=8)
        self.embed_dim = embed_dim

    def forward(self, features):
        projected = []
        for feat, proj in zip(features, self.proj_layers):
            feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
            projected.append(proj(feat))
        x = torch.cat(projected, dim=1)
        
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        split_sizes = [self.embed_dim] * len(features)
        outputs = torch.split(x, split_sizes, dim=1)
        outputs = [inv_proj(out) for out, inv_proj in zip(outputs, self.inv_proj_layers)]
        return [F.interpolate(out, size=feat.shape[2:]) for out, feat in zip(outputs, features)]
    
class PerspectivePlusUNet(nn.Module):
    def __init__(self, base_ch=32, in_ch=1, out_ch=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            BiPathResidualBlock(in_ch, base_ch),
            nn.Sequential(
                nn.MaxPool2d(2),
                BiPathResidualBlock(base_ch, base_ch*2),
                ENLSA(base_ch*2)
            ),
            nn.Sequential(
                nn.MaxPool2d(2),
                BiPathResidualBlock(base_ch*2, base_ch*4),
                ENLSA(base_ch*4)
            )
        ])
        self.scsi = SpatialCrossScaleIntegrator(
            channels=[base_ch, base_ch*2, base_ch*4]
        )
        self.decoder = nn.ModuleList([
            UpSampleBlock(base_ch*4, base_ch*2),
            UpSampleBlock(base_ch*2, base_ch),
            nn.Conv2d(base_ch, out_ch, 1)
        ])

    def forward(self, x):
        features = []
        for stage in self.encoder:
            x = stage(x)
            features.append(x)
        enhanced = self.scsi(features)
        for i, layer in enumerate(self.decoder):
            x = layer(x + enhanced[-(i+1)])
        return x

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