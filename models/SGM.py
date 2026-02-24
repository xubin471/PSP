import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# =================================================================
# 1. 工具函数：提取 SDM 和 Hu 矩特征
# =================================================================
def get_sdm_features(mask):
    """
    Input: mask (H, W) numpy array, 0 or 1
    Output: feature_vector (Length = 3 [SDM] + 15 [Fourier] = 18)
    """
    # 定义傅里叶描述子保留的低频分量个数
    NUM_FOURIER = 15
    TOTAL_DIM = 3 + NUM_FOURIER

    if np.sum(mask) == 0:
        return np.zeros(TOTAL_DIM).astype(np.float32)

    h_img, w_img = mask.shape
    img_diagonal = np.sqrt(h_img ** 2 + w_img ** 2) + 1e-6

    # ---------------------------------------------------------
    # Part A: SDM 特征 (3维) - 描述内部厚度和均匀度
    # ---------------------------------------------------------
    dist_map = distance_transform_edt(mask)
    d_vals = dist_map[mask == 1]

    if len(d_vals) == 0:
        sdm_feat = np.zeros(3)
    else:
        # 归一化：除以对角线，使其落入 [0, 1] 区间
        sdm_feat = np.array([np.max(d_vals), np.mean(d_vals), np.std(d_vals)]) / img_diagonal

    # ---------------------------------------------------------
    # Part B: 傅里叶描述子 (15维) - 描述轮廓形状 (低频)
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        fourier_feat = np.zeros(NUM_FOURIER)
    else:
        # 取最大的轮廓
        cnt = max(contours, key=cv2.contourArea)

        # 将轮廓坐标 (x, y) 转换为复数 z = x + iy
        # cnt shape: (N, 1, 2)
        cnt_pts = cnt[:, 0, :]
        contour_complex = np.empty(cnt_pts.shape[0], dtype=complex)
        contour_complex.real = cnt_pts[:, 0]
        contour_complex.imag = cnt_pts[:, 1]

        # 1. 离散傅里叶变换 (DFT)
        fourier_result = np.fft.fft(contour_complex)

        # 2. 取模长 (Magnitude) -> 获得旋转不变性
        descriptors = np.abs(fourier_result)

        # 3. 尺度归一化 & 平移不变性
        # descriptors[0] 是直流分量(位置信息)，我们不需要 -> 丢弃
        # descriptors[1] 是基频分量(整体大小)，用它来做归一化
        if len(descriptors) > 1 and descriptors[1] > 0:
            # 除以基频分量，消除尺度影响
            descriptors = descriptors / descriptors[1]
            # 丢弃第一个分量(DC)
            descriptors = descriptors[1:]
        else:
            descriptors = np.zeros(NUM_FOURIER)

        # 4. 截断或填充
        # 我们只取前 NUM_FOURIER 个分量 (低频)，忽略后面的高频(噪声)
        if len(descriptors) < NUM_FOURIER:
            fourier_feat = np.pad(descriptors, (0, NUM_FOURIER - len(descriptors)))
        else:
            fourier_feat = descriptors[:NUM_FOURIER]

    # ---------------------------------------------------------
    # Part C: 拼接
    # ---------------------------------------------------------
    features = np.concatenate([sdm_feat, fourier_feat])

    # 安全截断，虽然傅里叶归一化后通常在 0-1 之间，SDM 也在 0-1 之间
    features = np.clip(features, -2.0, 2.0)

    return np.nan_to_num(features).astype(np.float32)

# =================================================================
# 2. 网络模块：轻量级 MLP
# =================================================================
class ShapeToChannelMLP(nn.Module):
    def __init__(self, input_feat_dim=64, hidden_dim=512, output_dim=512):
        super().__init__()
        # 改动：输入不再是 256*256，而是 pooled 之后的维度 (e.g., 8*8=64)
        self.linear1 = nn.Linear(input_feat_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [Channels, Pooled_Spatial_Dim]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ShapeProto(nn.Module):
    def __init__(self, emb_dim=512, protos_num=1, init_shape_dim=18):
        super().__init__()
        self.emb_dim = emb_dim
        self.protos_num = protos_num
        self.init_shape_dim = init_shape_dim

        self.shape_encoder = nn.Sequential(
            nn.Linear(init_shape_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb_dim),
        )

        # ===> 新增：Decoder (用于重构) <===
        self.shape_decoder = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, init_shape_dim) # 尝试还原回 13 维
        )

    def forward(self, msk):
        """
        Input: msk (B, 256, 256) or (B, 1, 256, 256) GPU Tensor
        Output: shape_proto (B, 512, 1) GPU Tensor
        """
        if msk.dim() == 4:
            msk = msk.squeeze(1)  # (B, H, W)

        batch_size = msk.shape[0]
        device = msk.device

        # 将 GPU Tensor 转为 CPU Numpy 进行 cv2/scipy 计算
        # 注意：这部分没有梯度，仅作为先验输入
        msk_np = msk.detach().cpu().numpy()

        batch_features = []
        for i in range(batch_size):
            # 逐个样本提取手工特征
            feat = get_sdm_features(msk_np[i])
            batch_features.append(feat)

        # (B, 13)
        batch_features = torch.tensor(np.array(batch_features)).float().to(device)

        # 归一化输入特征
        # batch_features = F.normalize(batch_features, dim=1)

        # Encoder: (B, 13) -> (B, 512*protos_num)
        encoded = self.shape_encoder(batch_features)
        encoded = F.normalize(encoded,dim=1)
        # Reshape to (B, 512, 1) assuming protos_num=1
        encoded = encoded.view(batch_size, self.emb_dim, self.protos_num)
        reconstructed = (self.shape_decoder(encoded.permute(0,2,1))).squeeze(1)
        rec_loss = F.smooth_l1_loss(reconstructed, batch_features.detach(), beta=0.1)

        return encoded, rec_loss


class SGM(nn.Module):
    def __init__(self):
        super().__init__()
        # 改动：使用 8x8 池化，输入维度由 65536 降为 64
        self.pool_size = 8
        self.shape_fts_generator = ShapeToChannelMLP(
            input_feat_dim=self.pool_size * self.pool_size,
            hidden_dim=512,
            output_dim=512
        )
        self.scalar = 20.0
        self.shape_proto_module = ShapeProto(emb_dim=512)

    def forward(self, sup_fts, sup_msk):
        """
        sup_fts: [B, 512, H, W]
        sup_msk: [B, H, W]
        """
        B, C, H, W = sup_fts.shape
        target_size = (256, 256)

        # 1. 统一尺寸
        if (H, W) != target_size:
            sup_fts = F.interpolate(sup_fts, size=target_size, mode="bilinear", align_corners=True)
            # Mask 使用 nearest 防止产生小数
            sup_msk = F.interpolate(sup_msk.unsqueeze(1).float(), size=target_size, mode="nearest").squeeze(1)

        # 确保 mask 是 0/1 且维度正确 [B, 1, 256, 256]
        if sup_msk.dim() == 3:
            sup_msk = sup_msk.unsqueeze(1)
        sup_msk = (sup_msk > 0.5).float()

        # 2. 提取 Mask 内的特征 (Masked Features)
        # [B, 512, 256, 256]
        positive_sup_fts = sup_fts * sup_msk

        # 3. 降维处理 (关键优化)
        # 不要直接 Flatten 256x256，先池化到 8x8
        # [B, 512, 8, 8]
        pooled_fts = F.adaptive_avg_pool2d(positive_sup_fts, (self.pool_size, self.pool_size))

        # Reshape: [B, 512, 64] -> Treat channels as items to embed
        # 我们希望为每个通道生成一个 Embedding，看看它是否符合 Shape Proto
        # MLP 需要输入: [B*512, 64] 或 [B, 512, 64]
        # 这里为了保持 Batch 维度，我们手动处理
        pooled_flat = pooled_fts.view(B, C, -1)  # [B, 512, 64]

        # MLP 处理:
        # 由于 PyTorch Linear 只作用于最后一维，我们可以直接传入 [B, 512, 64]
        # Linear(64, 512) 会作用于每个 Channel
        shape_sup_fts = self.shape_fts_generator(pooled_flat)  # Output: [B, 512, 512] (B, Channels, Emb_Dim)

        # 4. 获取几何形状先验 (Geometric Proto)
        # shape_proto: [B, 512, 1]
        shape_proto_vec, rec_loss = self.shape_proto_module(sup_msk)

        # 5. 计算相似度 (Channel Attention)
        # Normalize
        shape_sup_fts_norm = F.normalize(shape_sup_fts, dim=-1)  # [B, 512, 512]
        shape_proto_norm = F.normalize(shape_proto_vec, dim=1)  # [B, 512, 1] (Vector dimension is dim 1)
        # Wait, ShapeProto output is (B, 512, 1) where 512 is embedding dimension?
        # Let's check ShapeProto output: view(batch_size, emb_dim, protos_num) -> (B, 512, 1)
        # shape_sup_fts is (B, Channel=512, Emb=512)

        # 我们需要比较: 每个 Channel 的 Embedding vs 唯一的 Shape Proto
        # Shape Proto 应该 permute 成 [B, 1, 512] 以便广播
        shape_proto_norm = shape_proto_norm.permute(0, 2, 1)  # [B, 1, 512]

        # Dot product
        # [B, 512, 512] * [B, 1, 512] -> sum dim -1 -> [B, 512]
        sim_score = torch.sum(shape_sup_fts_norm * shape_proto_norm, dim=-1)  # [B, 512]

        # Sigmoid scaling
        sim_score = torch.sigmoid(sim_score * self.scalar)

        # 返回结果 (加上偏置 0.1)
        return sim_score + 0.2, rec_loss


# 测试代码
if __name__ == "__main__":
    model = SGM().cuda()
    # 模拟 Batch Size = 2
    dummy_fts = torch.randn(2, 512, 64, 64).cuda()
    dummy_msk = torch.randint(0, 2, (2, 256, 256)).float().cuda()  # 随机 0/1 Mask

    out , loss = model(dummy_fts, dummy_msk)
    print(f"Output shape: {out.shape} ---> loss:{loss}")  # 应该输出 [2, 512]