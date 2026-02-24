import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label, binary_fill_holes
# import mahotas
import numpy as np
import cv2
def keep_largest_component_by_cc(mask):
    """
    Args: mask:np (256 256)
    使用连通域分析保留最大区域
    """
    mask = (mask>0.5).astype(np.uint8)
    if mask.max() <= 1: mask = mask * 255

    # 1. 连通域分析
    # num_labels: 连通域数量 (包含背景)
    # labels: 标记后的图 (0是背景, 1是第一个区域...)
    # stats: 统计信息 [x, y, w, h, area]
    # centroids: 质心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # [边界情况] 只有背景
    if num_labels < 2:
        return np.zeros_like(mask)

    # 2. 找到最大区域的 label 索引
    # stats[i, cv2.CC_STAT_AREA] 是第 i 个区域的面积
    # 注意：label 0 是背景，一定要排除掉！
    # argsort 排序后，取最后一个就是最大面积的索引
    # stats[1:, 4] 取出除了背景外的所有面积
    # np.argmax 返回的是基于切片后的索引，所以最后要 +1
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 3. 生成新掩码
    new_mask = np.zeros_like(mask)
    new_mask[labels == max_label] = 1 # 或者 255

    return new_mask

# =================================================================
# 1. 工具函数：提取 SDM 和 Hu 矩特征
# =================================================================
def get_sdm_features(mask):
    """
    Input: mask (H, W) numpy array, 0 or 1
    Output: feature_vector (Length = 3 [SDM] + 15 [Fourier] = 18)
    """
    mask = mask>0

    # mask = keep_largest_component_by_cc(mask)
    # 定义傅里叶描述子保留的低频分量个数
    NUM_FOURIER = 15
    TOTAL_DIM = 3 + NUM_FOURIER + 3

    if np.sum(mask) == 0:
        return np.zeros(TOTAL_DIM).astype(np.float32)

    h_img, w_img = mask.shape
    img_diagonal = np.sqrt(h_img ** 2 + w_img ** 2)/4 + 1e-6

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
        # ------------------------------------------------------------
        # --- Feature C: Contour Properties ---
        # ------------------------------------------------------------
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)

            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            # [Fix 2] 长宽比取 Log，使其以 0 为中心 (1:1 -> 0)
            aspect_ratio = np.log(float(w_rect) / (h_rect + 1e-6) + 1e-6)
        else:
            compactness, solidity, aspect_ratio = 0, 0, 0

        contour_feat = np.array([compactness, solidity, aspect_ratio])

    # ---------------------------------------------------------
    # Part D: concat
    # ---------------------------------------------------------
    features = np.concatenate([sdm_feat, fourier_feat, contour_feat])


    features = np.clip(features, -2.0, 2.0)

    return np.nan_to_num(features).astype(np.float32)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=256,
        hidden_dim=512,
        dropout=0.2
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim*input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return (
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )




class ShapeProto(nn.Module):
    def __init__(self,emb_dim=512,protos_num=1,init_shape_dim=21):
        super().__init__()
        self.emb_dim = emb_dim
        self.protos_num = protos_num
        self.shape_encoder = nn.Sequential(
            nn.Linear(init_shape_dim,64),
            nn.ReLU(),
            nn.Linear(64,emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim,emb_dim*self.protos_num),
        )

    def forward(self,msk):
        """
        fts: (1 512 64 64)
        content_proto: glob proto (1 512)
        msk: (1 256 256)
        """
        shape_proto = self.shape_proto(msk)
        return shape_proto

    def shape_proto(self,msk):
        h,w = msk.shape[-2:]
        msk = msk.view(-1,h,w)[0]
        # h_indices , w_indices = torch.where(msk==1)
        # height = (h_indices.max() - h_indices.min()) / h
        # width = (w_indices.max() - w_indices.min()) / w
        # tf_values = get_shape_fts(msk.detach().cpu().numpy())
        low_shape = get_sdm_features(msk.detach().cpu().numpy())
        shape_proto = torch.tensor([[*list(low_shape)]]).to(msk.device).float()
        shape_proto = self.shape_encoder(shape_proto)
        return shape_proto #[1 512]


class SPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_fts_generator = MLP(input_dim=256,hidden_dim=512,dropout=0.2)
        self.scalar = 20.0
        self.shape_proto = ShapeProto()

    def forward(self, sup_fts, sup_msk):
        """

        :param sup_fts: [1 512 256 256]
        :param sup_msk: [1 256 256]
        :param shape_proto: [1 512]
        :return:
        """

        # =================================================
        sup_fts = F.interpolate(sup_fts,size=(256,256),mode="bilinear") #[1 512 256 256]
        sup_msk = (F.interpolate(sup_msk[None,...].float(),size=(256,256),mode="bilinear").squeeze(0)>0).int()
        # =================================================
        positive_sup_fts = (sup_fts * sup_msk[None,...]) #[1 512 256 256]
        c,h,w = positive_sup_fts.shape[-3:]
        positive_sup_fts = positive_sup_fts.reshape(c,-1)
        shape_sup_fts = self.shape_fts_generator(positive_sup_fts) #[512, 512] previous num of channels， latter : num of shape fts
        # =================================================
        shape_proto = self.shape_proto(sup_msk)
        # =================================================
        shape_sup_fts_norm = F.normalize(shape_sup_fts,dim=1)
        shape_proto_norm = F.normalize(shape_proto,dim=1)
        sim_score = torch.sum(shape_sup_fts_norm * shape_proto_norm,dim=-1)[:,None] #[512 1]
        sim_score = F.sigmoid(sim_score*self.scalar).permute(1,0) #[1 512]  (0-1)
        return sim_score+0.1, torch.zeros(1).to(sup_fts.device)