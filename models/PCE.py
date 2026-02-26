import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbedding(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim // 2),  
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.pos_merge = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),  
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(-4.0))
        nn.init.constant_(self.pos_merge[-1].weight, 0)
        nn.init.constant_(self.pos_merge[-1].bias, 0)

    def get_polar_grid(self, h, w, device):
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device),
                              indexing='ij')

        # normalize to -1 ~ 1
        x = (x - w / 2) / (w / 2)
        y = (y - h / 2) / (h / 2)
        # x = x / w
        # y = y / h

        rho = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x) / torch.pi  # normalize to -1 ~ 1

        # Stack: [H, W, 1]
        grid = torch.stack([rho, theta], dim=-1)
        return grid

    def generate_pos_emd(self, x):
        """
        x: [B_total, C, H, W]
        """
        B, _, H, W = x.shape
        #  [H, W, 1]
        grid = self.get_polar_grid(H, W, x.device)

        # MLP: [H, W, 1] -> [H, W, C]
        pos_emb = self.mlp(grid)

        #  [H, W, C] -> [1, C, H, W] -> [B, C, H, W]
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        return pos_emb

    def forward(self, x):
        """
        sup_fts: [B, C, H, W]
        qry_fts: [B, C, H, W]
        """

        pos = self.generate_pos_emd(x)  # [B, C, H, W]

        # Concat: Feature + Pos -> [B, 2C, H, W]
        x_pos = torch.cat([x, pos], dim=1)

        x_pos = self.pos_merge(x_pos)  # [B, C, H, W]
        return x_pos


class PCE(nn.Module):
    def __init__(self, in_channels=512, inter_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        if self.inter_channels == 0:
            self.inter_channels = 1

        self.q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        self.conv_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.pos_emd = PositionEmbedding()
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_pos = self.pos_emd(x)

        q = self.q(x_pos).view(batch_size, self.inter_channels, -1)
        q = q.permute(0, 2, 1)  # (B, H*W, C_inter)

        k = self.k(x_pos).view(batch_size, self.inter_channels, -1)  # (B, C_inter, H*W)

        v = self.v(x_pos).view(batch_size, self.in_channels, -1)
        v = v.permute(0, 2, 1)  # (B, H*W, C_in)

        # (B, H*W, C_inter) @ (B, C_inter, H*W) -> (B, H*W, H*W)
        attention = torch.matmul(q, k)
        scale = self.inter_channels ** -0.5
        attention = attention * scale

        # Softmax
        attention = F.softmax(attention, dim=-1)

        # (B, H*W, H*W) @ (B, H*W, C_in) -> (B, H*W, C_in)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.in_channels, height, width)
        out = self.conv_out(out)
        return self.gamma * out  + x
