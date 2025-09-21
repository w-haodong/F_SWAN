# 文件: operation/decoder.py
import torch
import torch.nn.functional as F

def _gather_feat(feat, ind):
    """根据一维索引，从一个扁平化的特征图中抓取特征。"""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    """将(B,C,H,W)的特征图变形，并根据索引抓取特征。"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

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

    for b in range(batch_size):
        for k in range(K):
            # 1. 找到當前熱圖的全局最大值
            max_score, max_ind = torch.max(heat_clone[b].view(-1), 0)
            max_y = (max_ind // width).item()
            max_x = (max_ind % width).item()

            # 2. 保存結果
            topk_scores[b, k] = max_score
            topk_inds[b, k] = max_ind
            topk_ys[b, k] = float(max_y)
            topk_xs[b, k] = float(max_x)

            # 3. "抹黑"（抑制）該點周圍的鄰域
            y_start = max(0, max_y - suppression_radius)
            y_end = min(height, max_y + suppression_radius + 1)
            x_start = max(0, max_x - suppression_radius)
            x_end = min(width, max_x + suppression_radius + 1)
            heat_clone[b, 0, y_start:y_end, x_start:x_end] = 0

    return topk_scores, topk_inds, topk_ys, topk_xs


class DecDecoder(object):
    """
    解碼器最終版：採用迭代抑制策略，並在解碼前屏蔽邊界偽影。
    """
    def decode(self, heat: torch.Tensor, reg: torch.Tensor, wavelet: torch.Tensor, 
               K: int = 17, 
               suppression_radius: int = 3) -> torch.Tensor: # 對於256的圖，半徑3或4比較合適
        
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
        reg_gathered = _tranpose_and_gather_feat(reg, inds)
        wavelet_pred_gathered = _tranpose_and_gather_feat(wavelet, inds)
        
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
        
        return detections