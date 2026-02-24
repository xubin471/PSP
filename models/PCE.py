import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbedding(nn.Module):
    """
    位置分支核心：将几何坐标映射为语义特征维度
    优化点：修复硬编码、支持动态Batch Size、规范化Grid生成
    """

    def __init__(self, feature_dim=512, hidden_dim=64):
        super().__init__()

        # 坐标映射 MLP
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim // 2),  # 中间层可以适当调整
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # 特征融合层：输入是 特征+位置 (1 * feature_dim) -> 输出 feature_dim
        self.pos_merge = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),  # 加上BN通常收敛更好
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
        )

        # 可学习的加权系数，初始化为较小值
        self.alpha = nn.Parameter(torch.tensor(-4.0))
        nn.init.constant_(self.pos_merge[-1].weight, 0)
        nn.init.constant_(self.pos_merge[-1].bias, 0)

    def get_polar_grid(self, h, w, device):
        # 使用 indexing='ij' 消除警告
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device),
                              indexing='ij')

        # 归一化到 -1 ~ 1
        x = (x - w / 2) / (w / 2)
        y = (y - h / 2) / (h / 2)
        # x = x / w
        # y = y / h

        # 转换为极坐标
        rho = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x) / torch.pi  # 归一化到 -1 ~ 1

        # Stack: [H, W, 1]
        grid = torch.stack([rho, theta], dim=-1)
        return grid

    def generate_pos_emd(self, x):
        """
        x: [B_total, C, H, W]
        """
        B, _, H, W = x.shape
        # 生成网格 [H, W, 1]
        grid = self.get_polar_grid(H, W, x.device)

        # MLP 映射: [H, W, 1] -> [H, W, C]
        pos_emb = self.mlp(grid)

        # 调整维度: [H, W, C] -> [1, C, H, W] -> [B, C, H, W]
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        return pos_emb

    def forward(self, x):
        """
        支持任意 Batch Size
        sup_fts: [B, C, H, W]
        qry_fts: [B, C, H, W]
        """

        # 生成位置编码
        pos = self.generate_pos_emd(x)  # [B, C, H, W]

        # Concat: Feature + Pos -> [B, 2C, H, W]
        x_pos = torch.cat([x, pos], dim=1)

        # 融合
        x_pos = self.pos_merge(x_pos)  # [B, C, H, W]
        return x_pos


class PCE(nn.Module):
    """
    一个改进版的自注意力模块 (Non-local block)
    """

    def __init__(self, in_channels=512, inter_channels=None):
        super().__init__()

        self.in_channels = in_channels
        # 如果未指定中间通道数，则默认为输入通道数的一半
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        # 确保中间通道数至少为1
        if self.inter_channels == 0:
            self.inter_channels = 1

        # 查询（Query）卷积
        self.q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # 键（Key）卷积
        self.k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # 值（Value）卷积
        self.v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # 用于将加权后的特征重新映射回原始输入通道数
        self.conv_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # gamma 是一个可学习的参数，用于缩放注意力模块的输出
        # 初始化为0，使得网络在初始阶段更依赖于原始特征
        self.gamma = nn.Parameter(torch.zeros(1))

        self.pos_emd = PositionEmbedding()
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_pos = self.pos_emd(x)

        # 生成 Q, K, V
        # q 和 k 的通道数被压缩，以减少计算量
        q = self.q(x_pos).view(batch_size, self.inter_channels, -1)
        q = q.permute(0, 2, 1)  # (B, H*W, C_inter)

        k = self.k(x_pos).view(batch_size, self.inter_channels, -1)  # (B, C_inter, H*W)

        v = self.v(x_pos).view(batch_size, self.in_channels, -1)
        v = v.permute(0, 2, 1)  # (B, H*W, C_in)

        # 计算注意力权重
        # (B, H*W, C_inter) @ (B, C_inter, H*W) -> (B, H*W, H*W)
        attention = torch.matmul(q, k)

        # 缩放
        scale = self.inter_channels ** -0.5
        attention = attention * scale

        # Softmax
        attention = F.softmax(attention, dim=-1)

        # 加权求和
        # (B, H*W, H*W) @ (B, H*W, C_in) -> (B, H*W, C_in)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.in_channels, height, width)
        # 通过一个1x1卷积进行特征变换
        out = self.conv_out(out)
        return self.gamma * out  + x



# 示例
if __name__ == '__main__':
    # 创建一个输入张量
    input_tensor = torch.randn(1, 512, 64, 64)  # (batch_size, channels, height, width)

    # 改进的自注意力模块
    improved_sa = SelfAttention(in_channels=512)
    output_improved = improved_sa(input_tensor)
    print("改进模块输出尺寸:", output_improved.shape)