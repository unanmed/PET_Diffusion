import torch.nn as nn
from pytorch_msssim import ssim

class MSESSIMLoss(nn.Module):
    def __init__(self, alpha=0.8, ssim_window_size=11):
        """
        组合损失函数: MSE + SSIM

        参数:
        - alpha: MSE 与 SSIM 损失的权重比 (alpha 越高，MSE 占比越大)
        - ssim_window_size: SSIM 计算的窗口大小，通常为 11 或 7
        """
        super(MSESSIMLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim_window_size = ssim_window_size

    def forward(self, output, target):
        # MSE 损失
        mse_loss = self.mse(output, target)
        
        # SSIM 损失 (1 - SSIM 越小越好)
        ssim_loss = 1 - ssim(output, target, data_range=1, size_average=True, win_size=self.ssim_window_size)
        
        # 组合损失 (线性加权)
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
        return combined_loss