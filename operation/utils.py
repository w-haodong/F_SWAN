import os
import pywt
import cv2
import torch
import numpy as np

# --------------------------------------------------------------------------
# 评估所需函数
# --------------------------------------------------------------------------
def reconstruct_contour_from_wavelet_np(wavelet_coeffs_flat, center, wavelet_type, level, coeffs_len_per_axis,
                                        num_dense_points):
    """从预测的小波系数和中心点，重建轮廓 (NumPy版本)"""
    coeffs_x_flat, coeffs_y_flat = wavelet_coeffs_flat[:coeffs_len_per_axis], wavelet_coeffs_flat[coeffs_len_per_axis:]
    dummy_coeffs_layout = pywt.wavedec(np.zeros(num_dense_points), wavelet_type, level=level)

    def unflatten(coeffs_flat, layout):
        coeffs_list, start_idx = [], 0
        for c in layout:
            end_idx = start_idx + len(c);
            coeffs_list.append(coeffs_flat[start_idx:end_idx]);
            start_idx = end_idx
        return coeffs_list

    x_rel = pywt.waverec(unflatten(coeffs_x_flat, dummy_coeffs_layout), wavelet_type)
    y_rel = pywt.waverec(unflatten(coeffs_y_flat, dummy_coeffs_layout), wavelet_type)
    return np.stack([x_rel, y_rel], axis=1)[:num_dense_points] + center


def get_corners_from_contour(contour):
    """从密集轮廓点计算最小面积包围矩形的四个角点，并按[tl, tr, bl, br]顺序排序"""
    rect = cv2.minAreaRect(contour.astype(np.float32))
    box_points = cv2.boxPoints(rect)
    sorted_by_x = sorted(box_points, key=lambda p: p[0])
    left_points = sorted(sorted_by_x[:2], key=lambda p: p[1])
    right_points = sorted(sorted_by_x[2:], key=lambda p: p[1])
    tl, bl, tr, br = left_points[0], left_points[1], right_points[0], right_points[1]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def resort_gt_landmarks(gt_landmarks_original):
    """将从dataset加载的GT角点从 [tl, tr, br, bl] 顺序重排为 [tl, tr, bl, br] 顺序"""
    resorted_landmarks = gt_landmarks_original.copy()
    for i in range(0, len(gt_landmarks_original), 4):
        bl_point, br_point = gt_landmarks_original[i + 3].copy(), gt_landmarks_original[i + 2].copy()
        resorted_landmarks[i + 2], resorted_landmarks[i + 3] = bl_point, br_point
    return resorted_landmarks


# --------------------------------------------------------------------------
# 训练所需函数
# --------------------------------------------------------------------------
def reconstruct_contour_from_wavelet_np(
    wavelet_coeffs_flat: np.ndarray, 
    center: np.ndarray,
    wavelet_type: str,
    level: int,
    coeffs_len_per_axis: int,
    num_dense_points: int):
    """
    【新增】从预测的小波系数和中心点，批量重建轮廓 (NumPy版本，用于可视化)。
    """
    # 1. 拆分x和y的系数
    coeffs_x_flat = wavelet_coeffs_flat[:coeffs_len_per_axis]
    coeffs_y_flat = wavelet_coeffs_flat[coeffs_len_per_axis:]
    
    # 2. 通过一个哑元信号获取系数的长度结构
    dummy_signal = np.zeros(num_dense_points)
    dummy_coeffs_layout = pywt.wavedec(dummy_signal, wavelet_type, level=level)
    
    # 3. 根据长度结构，将一维向量恢复为pywt所需的列表格式
    def unflatten_coeffs(coeffs_flat, layout):
        coeffs_list = []
        start_idx = 0
        for c in layout:
            end_idx = start_idx + len(c)
            coeffs_list.append(coeffs_flat[start_idx:end_idx])
            start_idx = end_idx
        return coeffs_list

    coeffs_x_list = unflatten_coeffs(coeffs_x_flat, dummy_coeffs_layout)
    coeffs_y_list = unflatten_coeffs(coeffs_y_flat, dummy_coeffs_layout)

    # 4. 执行逆小波变换
    x_reconstructed_relative = pywt.waverec(coeffs_x_list, wavelet_type)
    y_reconstructed_relative = pywt.waverec(coeffs_y_list, wavelet_type)

    # 5. 加上中心点偏移，并堆叠成点云
    reconstructed_contour = np.stack(
        [x_reconstructed_relative, y_reconstructed_relative], axis=1
    )[:num_dense_points] + center

    return reconstructed_contour