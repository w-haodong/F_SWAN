import os
import math
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat
import pywt
import matplotlib.pyplot as plt

# 从本地模块导入辅助函数
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from operation import transform

# -----------------------------------------------------------------------------
# 1. 核心变换函数 (小波描述子)
# -----------------------------------------------------------------------------
def calculate_wavelet_descriptors(contour, wavelet='db4', level=4):
    """
    使用离散小波变换(DWT)为2D轮廓计算一维描述子。
    """
    x_signal = contour[:, 0]
    y_signal = contour[:, 1]
    coeffs_x_list = pywt.wavedec(x_signal, wavelet, level=level)
    coeffs_y_list = pywt.wavedec(y_signal, wavelet, level=level)
    coeffs_x_flat = np.concatenate(coeffs_x_list)
    coeffs_y_flat = np.concatenate(coeffs_y_list)
    return np.concatenate([coeffs_x_flat, coeffs_y_flat])

def reconstruct_from_wavelet_coeffs(wavelet_coeffs_flat, wavelet, level, original_len, coeffs_len_per_axis):
    """
    从小波系数向量逆变换回轮廓点，用于可视化验证。
    """
    coeffs_x_flat = wavelet_coeffs_flat[:coeffs_len_per_axis]
    coeffs_y_flat = wavelet_coeffs_flat[coeffs_len_per_axis:]
    
    dummy_coeffs_layout = pywt.wavedec(np.zeros(original_len), wavelet, level=level)
    
    coeffs_x_list = []
    start_index = 0
    for c in dummy_coeffs_layout:
        end_index = start_index + len(c)
        coeffs_x_list.append(coeffs_x_flat[start_index:end_index])
        start_index = end_index

    coeffs_y_list = []
    start_index = 0
    for c in dummy_coeffs_layout:
        end_index = start_index + len(c)
        coeffs_y_list.append(coeffs_y_flat[start_index:end_index])
        start_index = end_index

    x_reconstructed = pywt.waverec(coeffs_x_list, wavelet)
    y_reconstructed = pywt.waverec(coeffs_y_list, wavelet)

    return np.stack([x_reconstructed, y_reconstructed], axis=1)[:original_len]

# -----------------------------------------------------------------------------
# 2. 几何与辅助函数
# -----------------------------------------------------------------------------
def generate_sharp_contour_from_corners(pts_4, num_points=128):
    """
    从4个角点生成一个有锐利拐角的四边形密集轮廓。
    """
    pts_4_closed = np.vstack([pts_4, pts_4[0]])
    polygon_pts = np.empty((0, 2), dtype=np.float32)
    points_per_side = num_points // 4
    for i in range(4):
        segment = np.linspace(pts_4_closed[i], pts_4_closed[i+1], points_per_side, endpoint=False)
        polygon_pts = np.vstack([polygon_pts, segment])
    return polygon_pts

def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(br)
        boxes.append(bl)
    return np.asarray(boxes, np.float32)

# -----------------------------------------------------------------------------
# 3. PyTorch Dataset 主类
# -----------------------------------------------------------------------------
class BaseDataset(data.Dataset):
    def __init__(self, args, phase, 
                 domain='source',       # 'source' 或 'target'
                 labeled_subset=True, eval_data = None):  # True: 加载有标签的; False: 加载无标签的
        super(BaseDataset, self).__init__()
        self.args = args
        self.phase = phase
        self.domain = domain
        self.eval_data = eval_data
        self.input_h = self.args.input_h
        self.input_w = self.args.input_w
        self.down_ratio = self.args.down_ratio
        
        # 确保 self.labeled_subset 被正确赋值
        self.labeled_subset = labeled_subset

        if phase == 'train':
            self.img_dir = os.path.join(args.data_dir, self.phase, domain, 'images')
            self.label_dir = os.path.join(args.data_dir, self.phase, domain, 'labels')
        elif phase == 'val':
            self.img_dir = os.path.join(args.data_dir, self.phase, 'images')
            self.label_dir = os.path.join(args.data_dir, self.phase, 'labels')
        else:
            self.img_dir = os.path.join(args.data_dir, self.eval_data, 'images')
            self.label_dir = os.path.join(args.data_dir, self.eval_data, 'labels')

        all_image_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        temp_img_ids = []
        if self.phase == 'train' and self.domain == 'target':
            for fname in all_image_files:
                label_fname = os.path.splitext(fname)[0] + '.jpg.mat' 
                label_path = os.path.join(self.label_dir, label_fname)
                has_label = os.path.exists(label_path)
                
                if self.labeled_subset and has_label:
                    temp_img_ids.append(fname)
                elif not self.labeled_subset and not has_label:
                    temp_img_ids.append(fname)
        else:
            temp_img_ids = all_image_files

        self.img_ids = temp_img_ids
        
        if not self.img_ids:
            print(f"Warning: No images found for phase='{phase}', domain='{domain}', labeled_subset={self.labeled_subset}. Check folder: {self.img_dir}")

        # 小波变换
        self.wavelet_type = self.args.wavelet_type
        self.wavelet_level = self.args.wavelet_level
        self.num_dense_points = self.args.num_dense_points
        self.coeffs_len_per_axis = self.args.coeffs_len_per_axis
        self.total_coeffs_len = self.coeffs_len_per_axis * 2
        
        self.vis_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_wavelet_gt_check')
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

    def load_image(self, index):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        return image

    def load_gt_pts(self, annopath):
        pts = loadmat(annopath)['p2']
        pts = rearrange_pts(pts)
        return pts

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_dir, os.path.splitext(img_id)[0] + '.jpg.mat')

    def load_annotation(self, index):
        img_id = self.img_ids[index]
        annoFolder = self.load_annoFolder(img_id)
        pts = self.load_gt_pts(annoFolder)
        return pts

    # ---------------------- 新的 letterbox 几何预处理 ---------------------- #
    def _letterbox(self, image: np.ndarray, pts: np.ndarray | None):
        """
        固定画布 letterbox：
          - 画布大小：self.input_h x self.input_w
          - scale = min(input_h/H, input_w/W)
          - 图像按该比例缩放，不超过画布
          - 居中贴到画布上，剩余补 0
          - 点坐标同步做 scale + (left, top) 平移
        """
        H, W = image.shape[:2]
        T_h, T_w = int(self.input_h), int(self.input_w)

        # 等比例缩放因子
        scale = min(float(T_h) / float(H), float(T_w) / float(W))
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))

        # 缩放图像
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 画布 & 居中
        canvas = np.zeros((T_h, T_w, 3), dtype=image.dtype)
        top = (T_h - new_h) // 2
        left = (T_w - new_w) // 2
        canvas[top: top + new_h, left: left + new_w] = img_resized

        # 变换点
        if pts is None or pts.size == 0:
            out_pts = None
        else:
            t = pts.astype(np.float32).copy()
            t[:, 0] = t[:, 0] * scale + left
            t[:, 1] = t[:, 1] * scale + top
            out_pts = t

        meta = {
            "orig_h": float(H),
            "orig_w": float(W),
            "input_h": float(T_h),
            "input_w": float(T_w),
            "scale": float(scale),
            "top": float(top),
            "left": float(left),
        }
        return np.ascontiguousarray(canvas), out_pts, meta

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image_ori = self.load_image(index)

        pts_ori = None
        has_label = False
        label_path = self.load_annoFolder(img_id)
        
        if os.path.exists(label_path):
            try:
                pts_ori = self.load_gt_pts(label_path)
                has_label = True
            except Exception as e:
                print(f"Warning: Could not load or process label for {img_id}. Error: {e}")

        # === 用新的 letterbox 把原图和标注统一映射到固定画布 (input_h, input_w) ===
        image_lb, pts_lb, meta = self._letterbox(image_ori, pts_ori)
        pad_left = int(meta["left"])  # 保留一个横向 pad 信息，兼容原先 'pad' 字段

        if self.phase == 'test':
            if pts_lb is None:
                images, _ = processing_test(
                    image=image_lb, 
                    input_h=self.input_h, 
                    input_w=self.input_w, 
                    pts=None
                )
                return {'input': images, 'img_id': img_id}
            else:
                images, pts_lb_scaled = processing_test(
                    image=image_lb, 
                    input_h=self.input_h, 
                    input_w=self.input_w, 
                    pts=pts_lb
                )
                return {
                    'input': images,
                    'img_id': img_id,
                    'pad': np.array(pad_left),      # 仍然返回一个 pad（现在是 left）
                    'pts_pad': np.array(pts_lb_scaled)
                }
        else:
            # 无论是否有标签，都进行数据增强，以支持一致性学习
            aug_label = (self.phase == 'train' and has_label)
            
            out_image, pts_2 = processing_train(
                image=image_lb,
                pts=pts_lb,
                image_h=self.input_h,
                image_w=self.input_w,
                down_ratio=self.down_ratio,
                aug_label=aug_label,
                domain=self.domain
            )
            
            # 统一返回结构一致的字典
            if has_label:
                # 如果有标签，调用GT生成函数
                data_dict = generate_ground_truth(
                    image_tensor=out_image,
                    pts_2=pts_2,
                    image_h=self.input_h // self.down_ratio,
                    image_w=self.input_w // self.down_ratio,
                    img_id=img_id,
                    vis_dir=self.vis_dir,
                    num_dense_points=self.num_dense_points,
                    wavelet_type=self.wavelet_type,
                    wavelet_level=self.wavelet_level,
                    coeffs_len_per_axis=self.coeffs_len_per_axis,
                    total_coeffs_len=self.total_coeffs_len,
                    down_ratio=self.down_ratio
                )
                return data_dict
            else:
                # 如果没有标签，手动创建包含占位符的字典
                output_h = self.input_h // self.down_ratio
                output_w = self.input_w // self.down_ratio
                NUM_VERTEBRAE = 17 # Or self.args.K

                return {
                    'input': torch.from_numpy(out_image),
                    'pts_gt': np.zeros((NUM_VERTEBRAE * 4, 2), dtype=np.float32),
                    'hm_gt': np.zeros((1, output_h, output_w), dtype=np.float32),
                    'ind_gt': np.zeros((NUM_VERTEBRAE), dtype=np.int64),
                    'reg_gt': np.zeros((NUM_VERTEBRAE, 2), dtype=np.float32),
                    'reg_mask_gt': np.zeros((NUM_VERTEBRAE), dtype=np.uint8),
                    'wavelet_gt': np.zeros((NUM_VERTEBRAE, self.total_coeffs_len), dtype=np.float32)
                }
            

    def __len__(self):
        return len(self.img_ids)

# -----------------------------------------------------------------------------
# 5. GT 生成主函数 
# -----------------------------------------------------------------------------
def generate_ground_truth(image_tensor, pts_2, image_h, image_w,
                                  img_id, vis_dir, num_dense_points, wavelet_type, wavelet_level,
                                  coeffs_len_per_axis, total_coeffs_len, down_ratio):
    """
    生成GT，并在预处理后的图像(out_image)上进行正确的坐标空间可视化。
    """
    # --- 1. 参数与初始化 ---
    NUM_VERTEBRAE = 17
    output_h, output_w = image_h, image_w
    VISUALIZE_THIS_SAMPLE = False

    hm = np.zeros((1, output_h, output_w), dtype=np.float32) 
    reg = np.zeros((NUM_VERTEBRAE, 2), dtype=np.float32)
    ind = np.zeros((NUM_VERTEBRAE), dtype=np.int64)
    reg_mask = np.zeros((NUM_VERTEBRAE), dtype=np.uint8)
    wavelet_gt = np.zeros((NUM_VERTEBRAE, total_coeffs_len), dtype=np.float32)
    
    # --- 2. 可视化准备 ---
    vis_image = None
    if VISUALIZE_THIS_SAMPLE:
        # 将输入的Tensor(C,H_in,W_in)转回为可绘制的OpenCV图像(H_in,W_in,C)
        vis_image_np = (image_tensor + 0.5) * 255.0
        vis_image_np = np.clip(vis_image_np, 0, 255).astype(np.uint8)
        vis_image_np = np.transpose(vis_image_np, (1, 2, 0))
        vis_image = cv2.cvtColor(vis_image_np, cv2.COLOR_RGB2BGR).copy()
        
        # 计算正确的下采样率，用于将特征图坐标放大回可视化图像的坐标
        down_ratio = vis_image.shape[0] / output_h

    # --- 3. 遍历每个椎体 ---
    if pts_2 is not None:
        for k in range(NUM_VERTEBRAE):
            # pts_k 是在特征图空间 
            pts_k = pts_2[k*4 : (k+1)*4, :]
            if pts_k.sum() < 1: continue

            # dense_contour, ct, contour_centered 都在特征图空间
            dense_contour = generate_sharp_contour_from_corners(pts_k, num_points=num_dense_points)
            ct = np.mean(dense_contour, axis=0)
            contour_centered = dense_contour - ct
            
            # wavelet_coeffs 是从特征图空间的坐标计算得来
            wavelet_coeffs = calculate_wavelet_descriptors(
                contour_centered, wavelet=wavelet_type, level=wavelet_level
            )
            
            if VISUALIZE_THIS_SAMPLE and vis_image is not None:
                reconstructed_centered = reconstruct_from_wavelet_coeffs(
                    wavelet_coeffs, wavelet_type, wavelet_level, num_dense_points, coeffs_len_per_axis
                )
                # reconstructed_contour 仍然在特征图空间
                reconstructed_contour = reconstructed_centered + ct
                
                # --- 在绘图前，将所有坐标放大回 out_image 的空间 ---
                original_contour_to_draw = (dense_contour * down_ratio).astype(np.int32)
                reconstructed_contour_to_draw = (reconstructed_contour * down_ratio).astype(np.int32)
                
                # 绘制：原始轮廓(蓝色粗线) vs 重建轮廓(红色细点)
                # 现在它们的坐标已经与 vis_image (即out_image) 的尺寸匹配
                cv2.polylines(vis_image, [original_contour_to_draw], isClosed=True, color=(255, 100, 0), thickness=2) # Blue
                for pt in reconstructed_contour_to_draw:
                    cv2.circle(vis_image, tuple(pt), 1, (0, 0, 255), -1) # Red

            # 填充GT张量的逻辑不变，它们都是基于特征图空间的坐标
            ct_int = ct.astype(np.int32)
            if not (0 <= ct_int[0] < output_w and 0 <= ct_int[1] < output_h): continue
            reg_mask[k] = 1
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            wavelet_gt[k, :] = wavelet_coeffs / float(output_w)

            min_coords, max_coords = np.min(pts_k, axis=0), np.max(pts_k, axis=0)
            bbox_h, bbox_w = max(1, max_coords[1] - min_coords[1]), max(1, max_coords[0] - min_coords[0])
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius_scale_factor = 0.7 
            final_radius = max(0, int(radius * radius_scale_factor))
            draw_umich_gaussian(hm[0, :, :], ct_int, final_radius)

    # --- 4. 保存可视化结果 ---
    if VISUALIZE_THIS_SAMPLE and vis_image is not None:
        output_path = os.path.join(vis_dir, f"{os.path.splitext(img_id)[0]}_GT_CHECK.jpg")
        cv2.imwrite(output_path, vis_image)
        save_heatmap(hm, os.path.join(vis_dir, f"{os.path.splitext(img_id)[0]}_GT_HT.jpg"))

    # --- 5. 返回最终的GT字典 ---
    return {
        'input': image_tensor,
        'pts_gt': pts_2*down_ratio,
        'hm_gt': hm,
        'ind_gt': ind, 
        'reg_gt': reg, 
        'reg_mask_gt': reg_mask, 
        'wavelet_gt': wavelet_gt
    }

def processing_test(image, input_h, input_w, pts=None):
    """
    對測試圖像進行處理，包括 resize、歸一化，並可選地對標註點進行同步縮放。

    现在传入的 image 已经是 letterbox 后的画布 (input_h, input_w)，
    这里保持原来的写法，resize 到相同大小不会改变几何关系。
    """
    # 1. 在 resize 之前，獲取原始圖像的尺寸
    ori_h, ori_w = image.shape[:2]

    # 2. 對圖像進行 resize（如果已經是 input_h,input_w，這一步只是原位操作）
    image_resized = cv2.resize(image, (input_w, input_h))
    
    scaled_pts = None
    # 3. 如果提供了標註點，則進行同步縮放
    if pts is not None:
        scaled_pts = pts.astype(np.float32).copy()
        scale_w = input_w / ori_w
        scale_h = input_h / ori_h
        scaled_pts[:, 0] *= scale_w
        scaled_pts[:, 1] *= scale_h

    # 4. 對 resize 後的圖像進行歸一化和維度變換
    out_image = image_resized.astype(np.float32) / 255.
    out_image = out_image - 0.5
    out_image = out_image.transpose(2, 0, 1)
    
    # 5. 返回處理後的圖像和（可能存在的）縮放後的標註點
    return torch.from_numpy(out_image), scaled_pts

# -----------------------------------------------------------------------------
# 6. 数据增强与变换主函数
# -----------------------------------------------------------------------------
def processing_train(image, pts, image_h, image_w, down_ratio, aug_label, domain):
    if domain == 'source':
        min_scale = 0.5
        max_scale = 1.0
    else:
        min_scale = 0.95
        max_scale = 1.0

    data_aug = {
        'train': transform.Compose([transform.ConvertImgFloat(),
                                    transform.PhotometricDistort(),
                                    #transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                    transform.RandomScale(scale_range=(min_scale, max_scale)),
                                    transform.RandomRotate(angle_range=(-15, 15), prob=0.5),
                                    transform.Equalize(),
                                    transform.RandomMirror_w(),
                                    transform.Resize(h=image_h, w=image_w)]),
        'val': transform.Compose([transform.ConvertImgFloat(),
                                  transform.Resize(h=image_h, w=image_w)])
    }
    if aug_label and pts is not None:
        out_image, pts = data_aug['train'](image.copy(), pts)
    else:
        out_image, pts = data_aug['val'](image.copy(), pts)

    out_image = np.clip(out_image, a_min=0., a_max=255.)
    out_image = np.transpose(out_image / 255. - 0.5, (2, 0, 1))
    
    if pts is not None:
        pts = rearrange_pts(pts)
        pts2 = transform.rescale_pts(pts, down_ratio=down_ratio)
    else:
        pts2 = None

    return np.asarray(out_image, np.float32), pts2

def save_heatmap(hm, output_path="heatmap.png"):
    hm_np = hm[0]
    eps = 1e-12
    log_hm = np.log(hm_np + eps)
    min_val, max_val = np.min(log_hm), np.max(log_hm)
    normalized = (log_hm - min_val) / (max_val - min_val + 1e-8)
    uint8_hm = (normalized * 255).astype(np.uint8)
    color_hm = cv2.applyColorMap(uint8_hm, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, color_hm)
