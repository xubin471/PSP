import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label, distance_transform_edt
import math
class PredLoss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion = nn.NLLLoss(ignore_index=255,weight=torch.FloatTensor([0.1, 1.0]).cuda())

    def forward(self,query_pred,query_labels):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        query_loss = self.criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                     1 - torch.finfo(torch.float32).eps)), query_labels.long())

        return query_loss
class L1Loss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion =nn.L1Loss().cuda()

    def forward(self,a,b):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        L1Loss = self.criterion(a,b)

        return L1Loss
class AlignLoss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def forward(self,query_pred,query_labels):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        align_loss = self.criterion(query_pred.log(), query_labels.long())
        return align_loss
class CoarseLoss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion = nn.NLLLoss(ignore_index=255,weight=torch.FloatTensor([0.1, 1.0]).cuda())

    def forward(self,query_pred,query_labels):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        eps = torch.finfo(torch.float32).eps
        log_prob = torch.log(torch.clamp(query_pred, eps, 1 - eps))
        coarse_loss = self.criterion(log_prob, query_labels.long())
        return coarse_loss

def one_hot(label, n_classes, requires_grad=True):
    """
    Args:
        label: (1, H, W)
        n_classes: 2
        requires_grad: True or False
    """
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]  # (1, H, W, 2)
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)  # (1, 2, H, W) bg_mask and fg_mask
    # one_hot_label = one_hot_label.permute(0,3,1,2)
    return one_hot_label  # (1 2 256 256)


class BoundaryLoss(nn.Module):

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            pred: the output from model (before softmax)
                shape (1, 2, H, W)
            gt: ground truth map
                shape (1, H, W)
        Return:
            boundary loss, averaged over mini-batch
        """

        n, c, _, _ = pred.shape

        # pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)  # (1, 2, H, W)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt  # (1, 2, H, W)

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred  # (1, 2, H, W)

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)  # (1, 2, H, W)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)  # (1, 2, H, W)

        # reshape
        gt_b = gt_b.view(n, c, -1)  # (1, 2, HW)
        pred_b = pred_b.view(n, c, -1)  # (1, 2, HW)
        gt_b_ext = gt_b_ext.view(n, c, -1)  # (1, 2, HW)
        pred_b_ext = pred_b_ext.view(n, c, -1)  # (1, 2, HW)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        """
        row = 5
        col = 2
        # plt.subplot(row, col,1)
        # plt.imshow(gt[0].cpu().detach().numpy())
        # plt.title("gt")

        plt.subplot(row, col,1)
        plt.imshow(one_hot_gt[0][0].cpu().detach().numpy())
        plt.title("gt ont hot 0")

        plt.subplot(row, col,2)
        plt.imshow(one_hot_gt[0][1].cpu().detach().numpy())
        plt.title("gt ont hot 1")

        plt.subplot(row, col,3)
        plt.imshow(gt_b[0][0].cpu().detach().numpy())
        plt.title("gt border 0")

        plt.subplot(row, col,4)
        plt.imshow(gt_b[0][1].cpu().detach().numpy())
        plt.title("gt border 1")

        plt.subplot(row, col,5)
        plt.imshow(pred_b[0][0].cpu().detach().numpy())
        plt.title("pred_b border 0")

        plt.subplot(row, col,6)
        plt.imshow(pred_b[0][1].cpu().detach().numpy())
        plt.title("pred_b border 1")

        plt.subplot(row, col,7)
        plt.imshow(gt_b_ext[0][0].cpu().detach().numpy())
        plt.title("gt_b_ext border 0")

        plt.subplot(row, col,8)
        plt.imshow(gt_b_ext[0][1].cpu().detach().numpy())
        plt.title("gt_b_ext border 1")

        plt.subplot(row, col,9)
        plt.imshow(pred_b_ext[0][0].cpu().detach().numpy())
        plt.title("pred_b_ext border 0")

        plt.subplot(row, col,10)
        plt.imshow(pred_b_ext[0][1].cpu().detach().numpy())
        plt.title("pred_b_ext border 1")


        plt.show()
        """
        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseConsistencyLoss(nn.Module):
    """
    通过惩罚像素对关系预测的错误，强制模型学习目标的内部结构一致性。
    本实现假设输入 pred_mask 是经过 Softmax/Sigmoid 后的概率值。
    """

    def __init__(self, size=32):
        super().__init__()
        self.size = size
        # 我们的输入是概率，所以选择 BCELoss
        self.criterion = nn.BCELoss().cuda()

    def forward(self, pred_mask, label):
        """
        Args:
            pred_mask: (B, 2, H, W) or (B, 1, H, W) - 概率值 [0, 1]
            label: (B, H, W) or (B, 1, H, W) - 标签值 {0, 1}
        """
        # 确保 label 和 pred_mask 有4个维度
        if label.dim() == 3:
            label = label.unsqueeze(1)

        # 1. 准备目标矩阵
        label_resized = F.interpolate(label.float(), size=(self.size, self.size), mode='bilinear', align_corners=True)
        label_resized = (label_resized > 0.5).float()

        batch_size = label_resized.shape[0]
        lb_flat = label_resized.view(batch_size, -1)
        target_matrix = torch.bmm(lb_flat.unsqueeze(2), lb_flat.unsqueeze(1))

        # 2. 准备预测概率矩阵
        pred_resized = F.interpolate(pred_mask, size=(self.size, self.size), mode='bilinear', align_corners=True)

        # 如果是双通道Softmax输出，只取前景通道
        if pred_resized.shape[1] == 2:
            fg_prob = pred_resized[:, 1, ...]
        else:  # 否则认为是单通道Sigmoid输出
            fg_prob = pred_resized[:, 0, ...]

        fg_prob_flat = fg_prob.view(batch_size, -1)
        pred_prob_matrix = torch.bmm(fg_prob_flat.unsqueeze(2), fg_prob_flat.unsqueeze(1))

        # 3. 计算损失
        # BCELoss 期望的输入是概率，我们的 pred_prob_matrix 正是概率
        # 为了数值稳定，在传入前加一个微小的 clamp 是一个好习惯
        loss = self.criterion(
            torch.clamp(pred_prob_matrix, min=1e-7, max=1 - 1e-7),
            target_matrix
        )

        return loss

class MskLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.criterion = nn.MSELoss()
        pass
    def forward(self,pred_msk,msk):
        loss = torch.zeros(1).to(msk.device)
        if msk.sum() == 0:
            return loss
        x1,x2,x3 = torch.where(msk==1)
        loss +=self.criterion(pred_msk[x1,x2,x3],msk[x1,x2,x3])
        return loss

class KLShapeLoss(nn.Module):
    """
    使用KL散度约束形状分布的损失函数.

    将输入的掩码视为2D概率分布，并计算它们之间的KL散度。
    """
    def __init__(self, symmetric=False, epsilon=1e-8):
        """
        Args:
            symmetric (bool): 是否使用对称的KL散度 (即Jensen-Shannon散度的变体)。
                              对称版本通常更稳定，因为它同时惩罚两个方向的差异。
            epsilon (float): 用于防止log(0)和除以0的数值稳定项。
        """
        super().__init__()
        self.symmetric = symmetric
        self.epsilon = epsilon
        # 'batchmean' 会计算每个样本的KL散度，然后取批次的平均值。
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def _normalize_to_distribution(self, mask):
        """将掩码归一化为概率分布."""
        # 计算每个掩码在空间维度上的总和
        mask_sum = torch.sum(mask, dim=(-2, -1), keepdim=True)
        # 归一化，加上epsilon防止除以0
        distribution = mask / (mask_sum + self.epsilon)
        return distribution

    def forward(self, query_pred, support_mask):
        """
        Args:
            query_pred (torch.Tensor): query预测掩码，形状 (B, 1, H, W)，值为0-1之间.
            support_mask (torch.Tensor): support前景掩码，形状 (B, 1, H, W)，值为0或1.
        """
        # 1. 将掩码归一化为概率分布 P 和 Q
        P = self._normalize_to_distribution(support_mask.float())
        Q = self._normalize_to_distribution(query_pred)

        # 2. 准备 KLDivLoss 的输入
        # Pytorch的kl_div期望的input是对数概率
        log_P = torch.log(P + self.epsilon)
        log_Q = torch.log(Q + self.epsilon)

        # 3. 计算KL散度
        # KL(P || Q)，衡量用Q近似P时损失的信息
        loss_p_q = self.kl_loss(log_Q, P)

        if not self.symmetric:
            return loss_p_q
        else:
            # KL(Q || P)，衡量用P近似Q时损失的信息
            loss_q_p = self.kl_loss(log_P, Q)
            # 返回对称化的损失，即两者之和的一半
            return (loss_p_q + loss_q_p) / 2
class DifferentiableBoundaryExtractor(nn.Module):
    """
    一个可微分的模块,用于从概率掩码中提取边界距离向量.

    特征向量是质心到N个预设方向上"最远"边界的距离.
    """

    def __init__(self, num_directions=32, sharpness=10.0):
        """
        初始化模块.
        Args:
            num_directions (int): 方向数量.
            sharpness (float): 控制softmax的锐度. 值越大, "soft max"越接近真实的"hard max".
                               这是一个可以调整的超参数.
        """
        super().__init__()
        self.num_directions = num_directions
        self.sharpness = sharpness
        # 预先计算32个方向的单位向量
        self.directions = self._create_direction_vectors(num_directions)

    def _create_direction_vectors(self, num_directions):
        """生成均匀分布在360度内的单位方向向量."""
        angles = torch.linspace(0, 2 * math.pi, steps=num_directions, dtype=torch.float32)
        # directions shape: [num_directions, 2]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return directions


    def forward(self, mask):
        """
        从输入的掩码中提取边界距离特征.
        Args:
            mask (torch.Tensor): 输入的预测掩码, 形状为 [B, 1, H, W], 值在[0, 1]之间.
        Returns:
            torch.Tensor: 边界距离向量, 形状为 [B, num_directions].
        """
        if mask.dim() != 4 or mask.size(1) != 1:
            raise ValueError("输入掩码的期望形状是 [B, 1, H, W]")

        batch_size, _, h, w = mask.shape
        device = mask.device

        # 将方向向量移动到与掩码相同的设备上
        self.directions = self.directions.to(device)

        # 1. 创建像素坐标网格
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                        torch.arange(w, device=device, dtype=torch.float32),
                                        indexing='ij')
        # coords 形状: [H*W, 2] -> [[y_0, x_0], [y_1, x_1], ...]
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)

        # 2. 准备掩码
        mask_flat = mask.view(batch_size, -1)
        # 归一化掩码,使其总和为1, 作为概率分布
        mask_normalized = mask_flat / (mask_flat.sum(dim=1, keepdim=True) + 1e-8)

        # 3. 计算可微分的质心 (均值)
        centroid = torch.matmul(mask_normalized, coords)  # 形状: [B, 2]

        # 4. 计算所有点相对于质心的坐标
        centered_coords = coords.unsqueeze(0) - centroid.unsqueeze(1)  # 形状: [B, H*W, 2]

        # 5. 计算所有点在所有方向上的投影距离
        # projections 形状: [B, H*W, num_directions]
        projections = torch.matmul(centered_coords, self.directions.t())

        # 6. 计算可微分的最大边界距离 (核心步骤)
        # 我们希望找到在每个方向上, 投影距离最大的那个点.
        # Softmax可以帮我们创建一个"软"的 argmax.
        # 关键: 我们将掩码的对数加到投影上. 这能确保只有在掩码区域内的点(mask > 0)
        # 才对soft argmax有贡献, 掩码外的点(mask -> 0, log(mask) -> -inf)会被抑制.
        # log_mask = torch.log(mask_flat.unsqueeze(-1) + 1e-10) + 1 # 形状: [B, H*W, 1]
        #
        # # 将log_mask广播加到projections上
        # # 这样, 投影距离远且掩码概率高的点, 会有更大的值
        # # weighted_projections = projections + log_mask
        # weighted_projections = F.relu(projections) * log_mask

        log_mask = torch.log(mask_flat.unsqueeze(-1) + 1e-10)  # 形状: [B, H*W, 1]

        # 注意: 这里是 "加法" 并且 "没有 relu"
        # 这样才能同时考虑所有方向(正向和负向)的边界
        weighted_projections = projections + log_mask

        # 在像素维度(dim=1)上计算softmax. sharpness参数让分布更尖锐.
        # softmax_weights 形状: [B, H*W, num_directions]
        softmax_weights = F.softmax(weighted_projections * self.sharpness, dim=1)

        # 使用这些权重对原始的(无偏置的)投影距离进行加权求和, 得到"软"的最大距离
        # boundary_distances 形状: [B, num_directions]
        boundary_distances = torch.sum(softmax_weights * projections, dim=1)

        return boundary_distances


class BoundaryShapeLoss(nn.Module):
    """
    计算预测掩码的边界距离向量和目标向量之间的损失.
    """

    def __init__(self, num_directions=32, sharpness=10.0):
        super().__init__()
        self.shape_extractor = DifferentiableBoundaryExtractor(
            num_directions=num_directions,
            sharpness=sharpness
        )
        # 使用 L1 Loss 来对比两个向量. L2 (MSELoss) 也是一个不错的选择.
        self.loss_fn = nn.L1Loss()

    def forward(self, pred_msk, sup_msk):
        """
        Args:
            pred_msk: (1 1 h w)
            sup_msk: (1 1 h w)
        Returns:
            torch.Tensor: 形状损失值.
        """



        # 从预测掩码中提取可微分的边界距离向量
        predicted_distance_vector = self.shape_extractor(pred_msk)
        # return predicted_distance_vector
        with torch.no_grad():
            sup_distance_vector = self.shape_extractor(sup_msk)
        # 计算预测向量和目标向量之间的L1损失
        loss = self.loss_fn(predicted_distance_vector, sup_distance_vector)
        # loss = 1-F.cosine_similarity(predicted_distance_vector,sup_distance_vector,dim=1)

        return loss
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_loss = PredLoss()
        self.boundary_loss = BoundaryLoss(theta0=3, theta=5)
        # self.topo_loss = TopoLoss()
        self.align_loss = AlignLoss()
        self.msk_loss = MskLoss()
        self.l1_loss = L1Loss()
        self.kl_shape_loss = KLShapeLoss()
        self.bd_shape_loss = BoundaryShapeLoss()
        self.pair_wise_consistency_loss = PairwiseConsistencyLoss()
        self.coarse_loss = CoarseLoss()



