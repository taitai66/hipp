import torch
import torch.nn as nn
from einops import rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv1d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # (B, 1, L)
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        x2 = torch.cat([x_avg, x_max], dim=1)  # (B, 2, L)
        sattn = self.sa(x2)  # (B, 1, L)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        reduced_dim = max(1, dim // reduction)
        self.ca = nn.Sequential(
            nn.Conv1d(dim, reduced_dim, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_dim, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)  # (B, C, 1)
        cattn = self.ca(x_gap)  # (B, C, 1)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv1d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, L = x.shape
        x = x.unsqueeze(dim=2)  # (B, C, 1, L)
        pattn1 = pattn1.unsqueeze(dim=2)  # (B, 1, 1, L)
        pattn1 = pattn1.expand(-1, C, -1, -1)  # (B, C, 1, L)
        x2 = torch.cat([x, pattn1], dim=2)  # (B, C, 2, L)
        x2 = rearrange(x2, 'b c t l -> b (c t) l')  # (B, 2C, L)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim_sar=5, dim_opt=1, reduction=4):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim_sar, reduction)
        self.pa = PixelAttention(dim_sar)
        self.conv = nn.Conv1d(dim_sar, dim_sar, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.opt_proj = nn.Conv1d(dim_opt, dim_sar, 1, bias=True)

    def forward(self, x, y):
        """
        Args:
            x: SAR feature tensor of shape (B, L, C)
            y: OPT feature tensor of shape (B, L)
        Returns:
            result: fused feature tensor of shape (B, C, L)
        """
        B, L, C = x.shape
        # 调整SAR特征的维度
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        # 调整OPT特征的维度
        if y.dim() == 2:
            y = y.unsqueeze(1)  # (B, L) -> (B, 1, L)
        elif y.dim() == 3 and y.shape[2] == 1:
            y = y.permute(0, 2, 1)  # (B, L, 1) -> (B, 1, L)

        # 确保维度正确
        assert x.shape[0] == B and x.shape[1] == 5 and x.shape[2] == L
        assert y.shape[0] == B and y.shape[1] == 1 and y.shape[2] == L

        # 投影OPT特征到相同的通道维度
        y = self.opt_proj(y)  # (B, 5, L)

        initial = x + y  # 现在维度匹配：(B, 5, L)
        cattn = self.ca(initial)  # (B, 5, 1)
        sattn = self.sa(initial)  # (B, 1, L)

        pattn1 = sattn + cattn  # 广播相加
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  # (B, 5, L)

        result = initial + pattn2 * x + (1 - pattn2) * y  # (B, 5, L)
        result = self.conv(result)  # (B, 5, L)

        return result