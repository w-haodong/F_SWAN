# 文件: operation/decoder.py
import torch
from .tensor_ops import gather_spatial_features

def _get_topk_with_iterative_suppression(heat: torch.Tensor, K: int, suppression_radius: int):
    """
    【核心新函數】
    通過迭代抑制的方式，在原始熱圖上提取K個空間分離的峰值點。
    """
    batch_size, _, height, width = heat.shape
    
    topk_scores = torch.zeros(batch_size, K, device=heat.device)
    topk_inds = torch.zeros(batch_size, K, dtype=torch.long, device=heat.device)
    topk_ys = torch.zeros(batch_size, K, device=heat.device)
    topk_xs = torch.zeros(batch_size, K, device=heat.device)
    
    # 為了不修改原始熱圖，創建一個副本
    heat_clone = heat.clone()

    offsets = torch.arange(-suppression_radius, suppression_radius + 1, device=heat.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
    dy, dx = dy.reshape(1, -1), dx.reshape(1, -1)
    flat = heat_clone.view(batch_size, -1)
    for k in range(K):
        # Keep sequential peak suppression and first-index tie breaking, while
        # processing the batch together and keeping all indices on the device.
        max_score, max_ind = flat.max(dim=1)
        max_y = torch.div(max_ind, width, rounding_mode='floor')
        max_x = max_ind.remainder(width)
        topk_scores[:, k] = max_score
        topk_inds[:, k] = max_ind
        topk_ys[:, k] = max_y
        topk_xs[:, k] = max_x
        ys = (max_y[:, None] + dy).clamp(0, height - 1)
        xs = (max_x[:, None] + dx).clamp(0, width - 1)
        flat.scatter_(1, ys * width + xs, 0.)

    return topk_scores, topk_inds, topk_ys, topk_xs


class DecDecoder(object):
    """
    解碼器最終版：採用迭代抑制策略，並在解碼前屏蔽邊界偽影。
    """
    def decode(self, heat: torch.Tensor, reg: torch.Tensor, wavelet: torch.Tensor, 
               K: int = 17, 
               suppression_radius: int = 3, return_indices: bool = False) -> torch.Tensor: # 對於256的圖，半徑3或4比較合適
        
        batch_size = heat.size(0)

        # 在解碼前“屏蔽”邊界，防止因邊界偽影導致的錯誤選點
        border_width = 2 
        heat_clone = heat.clone()
        heat_clone[:, :, :border_width, :] = 0
        heat_clone[:, :, -border_width:, :] = 0
        heat_clone[:, :, :, :border_width] = 0
        heat_clone[:, :, :, -border_width:] = 0
        
        # 步驟 1: 使用迭代抑制來獲取K個互相分離的峰值點
        scores, inds, ys, xs = _get_topk_with_iterative_suppression(heat_clone, K, suppression_radius)
        
        # 步驟 2: 提取對應的偏移量和小波係數
        reg_gathered = gather_spatial_features(reg, inds)
        wavelet_pred_gathered = gather_spatial_features(wavelet, inds)
        
        # 步驟 3: 計算最終的精確中心點座標
        final_xs = xs + reg_gathered[..., 0]
        final_ys = ys + reg_gathered[..., 1]
        
        # 步驟 4: 按y座標排序，確保椎體從上到下的解剖學順序
        _, sort_inds = torch.sort(final_ys, dim=1)
        
        scores_sorted = torch.gather(scores, 1, sort_inds)
        final_xs_sorted = torch.gather(final_xs, 1, sort_inds)
        final_ys_sorted = torch.gather(final_ys, 1, sort_inds)
        final_centers_sorted = torch.stack([final_xs_sorted, final_ys_sorted], dim=2)
        wavelet_pred_sorted = torch.gather(wavelet_pred_gathered, 1, sort_inds.unsqueeze(2).expand_as(wavelet_pred_gathered))
        
        # 步驟 5: 拼接成最終的detections張量
        detections = torch.cat([
            scores_sorted.unsqueeze(2), 
            final_centers_sorted, 
            wavelet_pred_sorted
        ], dim=2)
        
        if return_indices:
            return detections, torch.gather(inds, 1, sort_inds)
        return detections
