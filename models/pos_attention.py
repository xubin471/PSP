import torch
import torch.nn as nn
import torch.nn.functional as F


class CentroidPositionEmbedding(nn.Module):
    """
    基于重心的位置嵌入模块
    核心思想：利用 Support Mask 的重心作为坐标系原点，构建相对位置编码。
    """

    def __init__(self, feature_dim=512, hidden_dim=64):
        super().__init__()

        # 1. 坐标映射 MLP (输入2维坐标 -> 映射到 hidden -> 输出 feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # 2. 特征融合层
        self.pos_merge = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            # nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
        )

    def calculate_centroid(self, mask):
        """
        计算 Mask 的重心坐标 (normalized to -1 ~ 1)
        mask: [B, 1, H, W] (0 or 1)
        return: centroids [B, 2] (y, x)
        """
        B, _, H, W = mask.shape
        device = mask.device

        # 生成基础网格坐标
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device),
                                        indexing='ij')

        # 归一化到 0 ~ 1 方便计算，最后再映射回 -1 ~ 1
        y_grid = y_grid.float() / (H - 1)
        x_grid = x_grid.float() / (W - 1)

        # 扩展维度以匹配 Batch: [1, 1, H, W]
        y_grid = y_grid.view(1, 1, H, W)
        x_grid = x_grid.view(1, 1, H, W)

        # 计算总质量 (防止除以0)
        mass = mask.sum(dim=(2, 3), keepdim=True) + 1e-6

        # 计算加权和
        # center_y = sum(y * mask) / sum(mask)
        cy = (y_grid * mask).sum(dim=(2, 3), keepdim=True) / mass
        cx = (x_grid * mask).sum(dim=(2, 3), keepdim=True) / mass

        # [B, 1, 1, 1] -> [B, 2]
        # 将 0~1 映射回 -1~1
        cy = (cy.view(B) - 0.5) * 2
        cx = (cx.view(B) - 0.5) * 2

        return torch.stack([cy, cx], dim=1)

    def get_relative_grid(self, H, W, centroids):
        """
        生成相对于 centroids 的坐标网格
        centroids: [B, 2] (cy, cx) in range [-1, 1]
        return: grid [B, H, W, 2] (rho, theta) or (dy, dx)
        """
        B = centroids.shape[0]
        device = centroids.device

        # 1. 生成基础网格 (B, H, W)
        # 归一化到 -1 ~ 1
        y_range = torch.linspace(-1, 1, steps=H, device=device)
        x_range = torch.linspace(-1, 1, steps=W, device=device)

        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        # 扩展到 Batch: [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)

        # 2. 减去 Support Mask 的重心 (Offset)
        # centroids: [B, 2] -> cy: [B, 1, 1], cx: [B, 1, 1]
        cy = centroids[:, 0].view(B, 1, 1)
        cx = centroids[:, 1].view(B, 1, 1)

        # 得到相对坐标
        rel_y = grid_y - cy
        rel_x = grid_x - cx

        # 3. 转换为极坐标 (Polar Coordinates)
        # 极径 Rho
        rho = torch.sqrt(rel_x ** 2 + rel_y ** 2)
        # 极角 Theta (归一化到 -1 ~ 1)
        theta = torch.atan2(rel_y, rel_x) / torch.pi

        # Stack: [B, H, W, 2]
        grid = torch.stack([rho, theta], dim=-1)
        return grid

    def forward(self, qry_fts, sup_msk):
        """
        qry_fts: [B, C, H, W] - 原始 Query 特征
        sup_msk: [B, 1, H, W] - Support 掩码 (需下采样到与 qry_fts 相同大小)
        """
        B, C, H, W = qry_fts.shape

        # 1. 确保 Mask 尺寸匹配 (如果输入尺寸不一致，需要 interpolate)
        if sup_msk.shape[-2:] != (H, W):
            sup_msk = F.interpolate(sup_msk.float(), size=(H, W), mode='nearest')

        # 2. 计算 Support 重心 [B, 2]
        centroids = self.calculate_centroid(sup_msk)

        # 3. 生成以重心为原点的相对坐标网格 [B, H, W, 2]
        grid = self.get_relative_grid(H, W, centroids)

        # 4. MLP 映射位置编码
        # grid: [B, H, W, 2] -> mlp -> [B, H, W, C]
        pos_emb = self.mlp(grid)

        # 调整维度 [B, H, W, C] -> [B, C, H, W]
        pos_emb = pos_emb.permute(0, 3, 1, 2)

        # 5. 融合
        x_cat = torch.cat([qry_fts, pos_emb], dim=1)  # [B, 2C, H, W]


        return qry_fts + self.pos_merge(x_cat)


class PriorGuidedSelfAttention(nn.Module):
    """
    融入 Support 先验的 Self-Attention
    """

    def __init__(self, in_channels=512, inter_channels=None):
        super().__init__()

        # ... (常规参数定义与之前相同)
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2
        if self.inter_channels == 0: self.inter_channels = 1

        self.q = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # 使用新的 Position Embedding
        self.pos_emd = CentroidPositionEmbedding(feature_dim=in_channels)

    def forward(self, x, sup_msk):
        """
        x: Query Features [B, C, H, W]
        sup_msk: Support Masks [B, 1, H_mask, W_mask]
        """
        batch_size, channels, height, width = x.size()

        # 【核心修改】：传入 sup_msk 计算相对位置编码
        x_pos = self.pos_emd(x.clone(), sup_msk)

        # Q, K 使用带有“相对位置感知”的特征
        q = self.q(x_pos).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        k = self.k(x_pos).view(batch_size, self.inter_channels, -1)

        # V 保持原样 (或根据需要也使用 x_pos)
        v = self.v(x).view(batch_size, self.in_channels, -1).permute(0, 2, 1)

        # Attention
        attention = torch.matmul(q, k)
        attention = attention * (self.inter_channels ** -0.5)
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.in_channels, height, width)
        out = self.conv_out(out)

        return self.gamma * out + x


# ==========================================
# 测试代码
# ==========================================
if __name__ == '__main__':
    from utils import *
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟输入
    B, C, H, W = 1, 512, 64, 64
    qry_fts = torch.randn(B, C, H, W).to(device)

    # 模拟 Support Mask (假设原始尺寸是 256x256)
    # Batch 0: 重心在左上, Batch 1: 重心在右下
    sup_msk = torch.zeros(B, 1, 256, 256).to(device)
    sup_msk[0, :, 50:100, 50:100] = 1  # Top-Left
    # sup_msk[1, :, 150:200, 150:200] = 1  # Bottom-Right

    # 初始化模型
    model = PriorGuidedSelfAttention(in_channels=C).to(device)

    # 前向传播
    out = model(qry_fts, sup_msk)

    print(f"Output shape: {out.shape}")

    # 验证重心计算逻辑
    embedder = model.pos_emd
    msk_ds = F.interpolate(sup_msk, size=(H, W))
    centroids = embedder.calculate_centroid(msk_ds)
    print(f"\nCalculated Centroids (Normalized -1~1):")
    print(f"Batch 0 (Expect Top-Left < 0): {centroids[0].detach().cpu().numpy()}")
    print(f"Batch 1 (Expect Bottom-Right > 0): {centroids[1].detach().cpu().numpy()}")