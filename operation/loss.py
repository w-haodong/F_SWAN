import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D 

def orthogonality_loss(content_features, style_features):
    """
    【最终修正版 - 适配低显存/小批次】
    移除了批处理中心化步骤，以解决 batch_size=2 导致的数学锁定问题。
    损失值将不再依赖批次大小。
    """

    # 2. L2归一化
    #    此操作是逐样本进行的，不引入批次依赖
    content_features = F.normalize(content_features, p=2, dim=1)
    style_features = F.normalize(style_features, p=2, dim=1)
    
    # 3. 计算互相关矩阵
    correlation_matrix = torch.mm(content_features.t(), style_features)
    
    # 4. 返回弗罗贝尼乌斯范数的平方
    #    该值的大小现在将由 w_ortho 控制
    return torch.sum(correlation_matrix**2)


# --------------------------------------------------------------------------
# --- 1. 辅助函数 ---
# --------------------------------------------------------------------------
def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

# --------------------------------------------------------------------------
# --- 2. 可微分的轮廓重建函数 ---
# --------------------------------------------------------------------------
def reconstruct_contour_from_wavelet(
    wavelet_coeffs_flat: torch.Tensor, 
    centers: torch.Tensor, 
    idwt: IDWT1D, 
    wavelet_type: str, # <-- 【修正点】接收字符串参数
    coeffs_len_per_axis: int, 
    level: int,
    num_dense_points: int):
    
    n_objs = wavelet_coeffs_flat.size(0)
    if n_objs == 0:
        return torch.empty(0, num_dense_points, 2, device=wavelet_coeffs_flat.device)
    
    device = wavelet_coeffs_flat.device
    coeffs_x_flat = wavelet_coeffs_flat[:, :coeffs_len_per_axis]
    coeffs_y_flat = wavelet_coeffs_flat[:, coeffs_len_per_axis:]

    # 【修正点】直接使用传入的 wavelet_type 字符串，不再访问 idwt.wave
    dwt = DWT1D(wave=wavelet_type, J=level, mode='zero').to(device)
    dummy_signal = torch.zeros(n_objs, 1, num_dense_points, device=device)
    cA_dummy, cD_dummy_list = dwt(dummy_signal)
    
    def unflatten_coeffs(coeffs_flat, cA_shape, cD_shapes):
        cA_len = cA_shape[-1]
        cA = coeffs_flat[:, :cA_len].unsqueeze(1)
        cD_list = []
        start_idx = cA_len
        for cD_shape in cD_shapes:
            cD_len = cD_shape[-1]
            cD = coeffs_flat[:, start_idx : start_idx + cD_len].unsqueeze(1)
            cD_list.append(cD)
            start_idx += cD_len
        return cA, cD_list

    cA_x, cD_x_list = unflatten_coeffs(coeffs_x_flat, cA_dummy.shape, [c.shape for c in cD_dummy_list])
    cA_y, cD_y_list = unflatten_coeffs(coeffs_y_flat, cA_dummy.shape, [c.shape for c in cD_dummy_list])

    x_coords_relative = idwt((cA_x, cD_x_list)).squeeze(1)
    y_coords_relative = idwt((cA_y, cD_y_list)).squeeze(1)

    x_coords = centers[:, 0].unsqueeze(1) + x_coords_relative
    y_coords = centers[:, 1].unsqueeze(1) + y_coords_relative
    
    reconstructed_contours = torch.stack([x_coords, y_coords], dim=2)
    
    return reconstructed_contours

# --------------------------------------------------------------------------
# --- 3. 基础损失类 (保持不变) ---
# --------------------------------------------------------------------------
class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        if mask.sum() == 0: return torch.tensor(0., device=output.device)
        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
        return loss / (mask.sum() + 1e-4)

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, pred, gt):
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        neg_weights = torch.pow(1 - gt[neg_inds], 4)
        loss = 0
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
        num_pos = pos_inds.float().sum()
        if num_pos > 0:
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
            loss = -(pos_loss + neg_loss) / num_pos
        else:
            loss = -neg_loss.sum()
        return loss

# --------------------------------------------------------------------------
# --- 组合损失类 ---
# --------------------------------------------------------------------------
class LossAll(torch.nn.Module):
    def __init__(self, args):
        """
        损失函数分为“基础部分”和“可选轮廓部分”。
        - 基础损失 (权重固定为1.0): 包括热图损失和中心点回归损失。
        - 轮廓损失 (w_contour): 可选模块，用于消融实验。
          (内部固定 w_coeff=0.5, w_geom=0.5)
        
        Args:
            args: 命令行参数。
            w_contour (float): 轮廓损失的权重。设置为0可完全关闭此模块。
        """
        super(LossAll, self).__init__()
        self.args = args
        self.device = args.device

        # 内部固定轮廓子损失的权重
        self._w_coeff_internal = 0.5
        self._w_geom_internal = 0.5

        # --- 2. 初始化所有需要的损失函数和工具 ---
        # 基础损失函数是必需的
        self.L_hm = FocalLoss()
        self.L_off = RegL1Loss()

        self.idwt = IDWT1D(wave=args.wavelet_type, mode='zero').to(self.device)
        self.wavelet_level = args.wavelet_level
        self.num_dense_points = args.num_dense_points
        self.coeffs_len_per_axis = args.coeffs_len_per_axis
        self.wavelet_type = args.wavelet_type

    def forward(self, pr_decs, gt_batch, w_contour):
        device = pr_decs['hm'].device
        
        # --- 1. 计算基础损失 (必需项) ---
        hm_loss = self.L_hm(pr_decs['hm'], gt_batch['hm_gt'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask_gt'], gt_batch['ind_gt'], gt_batch['reg_gt'])
        
        # 组合基础损失，其总权重为1.0
        base_loss = hm_loss + off_loss
        
        # --- 2. 计算可选的轮廓损失 ---
        coeff_loss = torch.tensor(0., device=device)
        geom_loss = torch.tensor(0., device=device)
        
        mask = gt_batch['reg_mask_gt'] > 0
        num_objs = mask.sum()

        if num_objs > 0:
            # 提取共用数据
            pred_wavelet_gathered = _tranpose_and_gather_feat(pr_decs['wavelet'], gt_batch['ind_gt'])
            pred_wavelet_flat = pred_wavelet_gathered[mask]
            gt_wavelet_flat = gt_batch['wavelet_gt'][mask]
            output_w = self.args.input_w // self.args.down_ratio
            pred_wavelet_denorm = pred_wavelet_flat * output_w
            gt_wavelet_denorm = gt_wavelet_flat * output_w
            
            # 计算小波系数损失
            coeff_loss = F.l1_loss(pred_wavelet_denorm, gt_wavelet_denorm)
            
            # 提取几何损失所需中心点数据
            pred_centers_gathered = _tranpose_and_gather_feat(pr_decs['reg'], gt_batch['ind_gt'])
            pred_centers_flat = pred_centers_gathered[mask]
            gt_centers_flat = gt_batch['reg_gt'][mask]
            
            # 重建轮廓并计算几何损失
            pred_contours = reconstruct_contour_from_wavelet(
                pred_wavelet_denorm, pred_centers_flat, self.idwt, self.wavelet_type,
                self.coeffs_len_per_axis, self.wavelet_level, self.num_dense_points
            )
            with torch.no_grad():
                gt_contours = reconstruct_contour_from_wavelet(
                    gt_wavelet_denorm, gt_centers_flat, self.idwt, self.wavelet_type,
                    self.coeffs_len_per_axis, self.wavelet_level, self.num_dense_points
                )
            geom_loss = F.smooth_l1_loss(pred_contours, gt_contours)

        # 内部加权的轮廓总损失
        total_contour_loss = self._w_coeff_internal * coeff_loss + self._w_geom_internal * geom_loss

        # --- 3. 组合最终总损失 ---
        # 总损失 = 基础损失 + 加权的轮廓损失
        total_loss = base_loss + w_contour * total_contour_loss
        
        # --- 4. 准备统计数据字典 ---
        loss_stats = {
            'total_loss': total_loss.item(), 
            'hm_loss': hm_loss.item(), 
            'off_loss': off_loss.item(), 
            'contour_loss': total_contour_loss.item(), # 报告的是加权后的轮廓损失
            'coeff_loss': coeff_loss.item(), 
            'geom_loss': geom_loss.item()
        }
        
        return total_loss, loss_stats