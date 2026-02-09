import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionSpatialPriorLogit(nn.Module):
    def __init__(self, sigma_ratio=0.25, prior_weight=1.5):
        """
        Args:
            sigma_ratio: 控制高斯热图的范围
            prior_weight: 控制先验的强度。
                          注意：在Logit空间，这个值的物理意义不同于概率空间。
                          weight=2.0 意味着在先验最强处，将 Logit 值提升 2.0。
                          (例如从 0.5[logit 0] 提升到 0.88[logit 2])
        """
        super().__init__()
        self.sigma_ratio = sigma_ratio
        self.prior_weight = prior_weight
        self.eps = 1e-6  # 防止 log(0) 的微小数值

    def generate_gaussian_map(self, mask, H, W):
        """
        生成高斯热图 (保持不变)
        """
        B = mask.shape[0]
        device = mask.device

        y_grid = torch.arange(H, dtype=torch.float32, device=device)
        x_grid = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')

        priors = []
        sigma = min(H, W) * self.sigma_ratio

        for i in range(B):
            indices = torch.nonzero(mask[i] > 0.5)
            if len(indices) > 0:
                cy = indices[:, 0].float().mean()
                cx = indices[:, 1].float().mean()
                dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
                gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            else:
                gaussian = torch.zeros((H, W), device=device)
            priors.append(gaussian)

        return torch.stack(priors).unsqueeze(1)  # [B, 1, H, W]

    def forward(self, qry_prob, sup_msk):
        """
        Args:
            qry_prob: [B, 1, H, W] 预测概率 (0~1)
            sup_msk:  [B, H, W]    Support Mask
        """
        B, C, H, W = qry_prob.shape

        # 1. 对齐 Support Mask
        if sup_msk.shape[-2:] != (H, W):
            sup_msk = F.interpolate(sup_msk.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)

        # 2. 生成空间先验 (0~1)
        spatial_prior = self.generate_gaussian_map(sup_msk, H, W)

        # =======================================================
        # 核心步骤：Prob -> Logit -> Add -> Prob
        # =======================================================

        # A. 数值稳定性截断 (Clamp)
        # 概率不能是纯 0 或纯 1，否则 log 会变成 inf
        prob_clamped = torch.clamp(qry_prob, self.eps, 1.0 - self.eps)

        # B. 逆变换 (Inverse Sigmoid)
        # logit = log(p / (1-p))
        qry_logits = torch.log(prob_clamped / (1.0 - prob_clamped))

        # C. 注入先验
        # 逻辑：在先验指示的区域，增加 Logit 值 (即增加前景的置信度)
        # spatial_prior 是 0~1 的高斯图
        refined_logits = qry_logits + (spatial_prior * self.prior_weight)

        # D. 变换回概率 (Sigmoid)
        refined_prob = torch.sigmoid(refined_logits)

        return refined_prob
#
# import torch
# pred = torch.randn(1,1,256,256)
# msk = (torch.randn(1,256,256)>0.5).int()
#
# PositionSpatialPriorLogit()(pred,msk)