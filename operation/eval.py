import torch
import numpy as np
from models import scd_net
from operation import decoder, cobb_evaluate_base, utils
import os
import csv
import time
import pywt
import cv2
from datasets.dataset import BaseDataset,rearrange_pts 
from datasets import draw_points
import scipy.io as sio

class Network(object):
    def __init__(self, args, peft_encoder):
        torch.manual_seed(317)
        self.args = args
        self.img_show_dir = None
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.total_wavelet_coeffs = self.args.coeffs_len_per_axis * 2
        heads = {'hm': self.args.num_classes,
                 'reg': 2 * self.args.num_classes,
                 'wavelet': self.total_wavelet_coeffs}
        self.model = scd_net.SCDNet(peft_encoder=peft_encoder,
                                         heads=heads,
                                         vit_input_layer_indices=args.vit_input_layer_indices)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder()
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights_spinal from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model


    def eval_base(self, args):
        save_path =  args.work_dir+'/weights'
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dsets = self.dataset['spinal'](args=args, phase='test', eval_data=args.eval_data)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
        
        mat_save_dir = os.path.join(args.work_dir, 'mat_outputs_'+str(args.num_train_f_sample)+'_'+args.eval_data +'_'+str(args.num_code_run))
        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
            print(f"创建目录用于保存 .mat 文件: {mat_save_dir}")
        
        img_show_dir = os.path.join(args.work_dir, 'img_show_'+str(args.num_train_f_sample)+'_'+args.eval_data +'_'+str(args.num_code_run))
        if not os.path.exists(img_show_dir):
            os.makedirs(img_show_dir)
            print(f"创建目录用于保存 .jpg 文件: {img_show_dir}")
        self.img_show_dir = img_show_dir
        total_time = []
        landmark_dist = []
        base_pr_cobb_angles = []
        base_gt_cobb_angles = [] 
        pos_list_pr = []
        pos_list_gt = []
        vertebrae_labels = [
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
            'L1', 'L2', 'L3', 'L4', 'L5'
        ]

        # --- Added Code: Initialize Data Structures ---
        vertebra_distance_errors = {label: [] for label in vertebrae_labels}

        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            image_tensor = data_dict['input'].to(self.device)
            pts0 = []
            img_id = data_dict['img_id'][0]
            #print(f'processing {cnt + 1}/{len(data_loader)} image ... {img_id}')

            ori_image = dsets.load_image(cnt)
            ori_pad_image, pad = dsets._pad_image_width_to_match_height(ori_image)
            image_cobb_pr = ori_pad_image.copy()
            ori_image_regress = ori_pad_image.copy() #cv2.resize(ori_image, (args.input_w, args.input_h))
            ori_image_points = ori_pad_image.copy()

            with torch.no_grad():
                pr_decs,_,_,_ = self.model(image_tensor)
                # 解码器总是返回17个候选椎体
                hm = pr_decs['hm']
                reg = pr_decs['reg']
                wavelet = pr_decs['wavelet']
                detections = self.decoder.decode(
                    heat=hm, reg=reg, wavelet=wavelet
                ).cpu().numpy()[0]
            
            end_time = time.time()
            total_time.append(end_time - begin_time)

            all_pred_corners = []
            detections = sorted(detections, key=lambda d: d[2])  # 按y坐标排序

            for det in detections:
                contour_feat = utils.reconstruct_contour_from_wavelet_np(
                    det[3:] * (args.input_w / args.down_ratio), det[1:3], args.wavelet_type, args.wavelet_level,
                    args.coeffs_len_per_axis, args.num_dense_points)
                contour_img = contour_feat * args.down_ratio
                corners = utils.get_corners_from_contour(contour_img)
                all_pred_corners.append(corners)

            # 现在 all_pred_corners 列表里总是包含17个椎体的角点
            pr_landmarks = np.concatenate(all_pred_corners, axis=0)

            h_orig, w_orig = ori_pad_image.shape[:2]
            scale_w, scale_h = self.args.input_w / w_orig, self.args.input_h / h_orig
            pr_landmarks[:,0] /= scale_w
            pr_landmarks[:,1] /= scale_h

            # 计算预测Cobb角
            pr_ca1, pr_ca2, pr_ca3, pr_pos_list = cobb_evaluate_base.cobb_angle_calc(pr_landmarks, image_cobb_pr, is_train=False)
            base_pr_cobb_angles.append([pr_ca1, pr_ca2, pr_ca3])
            pos_list_pr.append(pr_pos_list)


            gt_landmarks_from_loader = data_dict['pts_pad'].numpy()[0]
            gt_landmarks = utils.resort_gt_landmarks(gt_landmarks_from_loader)

            dist = np.sqrt(np.sum((pr_landmarks - gt_landmarks) ** 2, axis=1))
            landmark_dist.extend(dist)

            pr_center = self.get_spine_centers(pr_landmarks)
            gt_center = self.get_spine_centers(gt_landmarks)
            for idx, vertebra_label in enumerate(vertebrae_labels):
                distance_error = np.sqrt(np.sum((pr_center[idx] - gt_center[idx]) ** 2))
                vertebra_distance_errors[vertebra_label].append(distance_error)
                pts_to_join = [
                        pr_center[idx], 
                        pr_landmarks[(idx)*4], 
                        pr_landmarks[(idx)*4 + 1], 
                        pr_landmarks[(idx)*4 + 2], 
                        pr_landmarks[(idx)*4 + 3]
                    ]
                pts0.append(np.concatenate(pts_to_join))

            gt_ca1, gt_ca2, gt_ca3, gt_pos_list = cobb_evaluate_base.cobb_angle_calc(gt_landmarks, None,
                                                                                        is_train=False)
            base_gt_cobb_angles.append([gt_ca1, gt_ca2, gt_ca3])
            pos_list_gt.append(gt_pos_list)

            # --- 关键改动 3: 在循环内保存每个图像的结果 ---
            # 将三个角度合并为一个数组
            pr_angles = np.array([pr_ca1, pr_ca2, pr_ca3])
            
            # 构建保存文件名
            mat_filename = f"pl_{img_id}.mat"
            mat_filepath = os.path.join(mat_save_dir, mat_filename)
            
            # 创建要保存到 .mat 文件的数据字典
            # 根据您的要求，关键点变量名为 pr_landmarks
            # 我们也把角度一起保存，变量名为 pr_angles
            pr_landmarks_ori = pr_landmarks.copy()
            pr_landmarks_ori[:, 0] -= pad
            save_dict = {
                'pr_landmarks': pr_landmarks_ori,
                'pr_angles': pr_angles
            }
            # 使用 scipy.io.savemat 保存数据
            sio.savemat(mat_filepath, save_dict)
            self.save_heatmap(hm, img_show_dir + '/hm_'+img_id)
            ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0, ori_image_regress, ori_image_points)
            base_name, _ = os.path.splitext(img_id)
            detections = self.decoder.decode(heat=hm.detach(), reg=reg.detach(), wavelet=wavelet.detach(), K=17)
            self.save_prediction_visualization(image_tensor=data_dict['input'], detections=detections.cpu().numpy()[0], gt_pts=data_dict.get('pts_pad', torch.empty(0)).cpu().numpy()[0], img_id=f"wavelet_gt_pr_"+base_name)
            self.save_prediction_visualization(image_tensor=data_dict['input'], detections=detections.cpu().numpy()[0], gt_pts=None, img_id=f"wavelet_pr_"+base_name)
            self.save_prediction_visualization_ori(image_tensor=data_dict['input'], detections=detections.cpu().numpy()[0], gt_pts=None, img_id=f"wavelet_pr_"+base_name)     
            # ======================== 诊断代码结束 ========================
            
            cv2.imwrite(os.path.join(img_show_dir, 'point_' + base_name + '.jpg'), ori_image_regress)
            cv2.imwrite(os.path.join(img_show_dir, 'point_only_'+img_id), ori_image_points)
            cv2.imwrite(os.path.join(img_show_dir, 'ca_pr_' + base_name + '.jpg'), image_cobb_pr)
            del ori_image_regress, ori_image_points

        base_pr_cobb_angles = np.asarray(base_pr_cobb_angles, np.float32)
        base_gt_cobb_angles = np.asarray(base_gt_cobb_angles, np.float32)

        pos_list_pr = np.asarray(pos_list_pr, np.float32)
        pos_list_gt = np.asarray(pos_list_gt, np.float32)

        # 计算各指标需要的基础数据
        base_out_abs = abs(base_pr_cobb_angles - base_gt_cobb_angles)
        # 原始差值（带符号）
        diff = base_pr_cobb_angles - base_gt_cobb_angles  # shape: (N, 3)

        # 将差值映射到 [-180, 180] 范围（圆统计最小差值）
        circular_diff = np.degrees(
            np.arctan2(np.sin(np.radians(diff)), np.cos(np.radians(diff)))
        )

        # 绝对值差
        base_out_abs = np.abs(circular_diff)

        # 1. CMAE（Circular Mean Absolute Error）——逐角度计算
        CMAE = np.mean(base_out_abs)

        # 2. ED（Euclidean Distance）——基于圆差值
        ED = np.mean(np.linalg.norm(base_out_abs, axis=1))

        # 3. MD（Manhattan Distance）——基于圆差值
        MD = np.mean(np.sum(base_out_abs, axis=1))

        # 4. CD（Chebyshev Distance）——基于圆差值
        CD = np.mean(np.max(base_out_abs, axis=1))

        # ============== 打印新增指标 ==============
        print('\n===== 新增评估指标（圆统计版本） =====')
        print('CMAE（圆形平均绝对误差）: {:.4f}°'.format(CMAE))
        print('ED（欧氏距离）: {:.4f}°'.format(ED))
        print('MD（曼哈顿距离）: {:.4f}°'.format(MD))
        print('CD（切比雪夫距离）: {:.4f}°'.format(CD))


        print('与真实角度误差评估(平均):')
        print('总角度差 ---- Base: {:.4f}度'.format(np.mean(base_out_abs)))
        print('PT角度差 ---- Base: {:.4f}度'.format(np.mean(base_out_abs[:, 0])))
        print('MT角度差 ---- Base: {:.4f}度'.format(np.mean(base_out_abs[:, 1])))
        print('TL角度差 ---- Base: {:.4f}度'.format(np.mean(base_out_abs[:, 2])))

        base_out_add = base_pr_cobb_angles + base_gt_cobb_angles
        base_term1 = np.sum(base_out_abs, axis=1)
        base_term2 = np.sum(base_out_add, axis=1)
        base_term2[base_term2==0] += 1e-5
        Base_SMAPE = np.mean(base_term1 / base_term2 * 100)

        print('总SMAPE ---- Base: {:.4f}'.format(Base_SMAPE))
        print('PT_SMAPE --- Base: {:.4f}'.format(self.SMAPE_single_angle(base_pr_cobb_angles[:,0], base_gt_cobb_angles[:,0])))
        print('MT_SMAPE --- Base: {:.4f}'.format(self.SMAPE_single_angle(base_pr_cobb_angles[:,1], base_gt_cobb_angles[:,1])))
        print('TL_SMAPE --- Base: {:.4f}'.format(self.SMAPE_single_angle(base_pr_cobb_angles[:,2], base_gt_cobb_angles[:,2])))

        print('与真实脊柱坐标误差评估(平均):')
        print('坐标点的均方误差（MSE）： {:.4f}像素'.format(np.mean(landmark_dist)))
        total_time = total_time[1:]
        #print('avg time is {}'.format(np.mean(total_time)))
        print('FPS: {:.4f}'.format(1./np.mean(total_time)))
        print("\n正在保存评估数据...")

        # 1. 定义保存路径
        save_dir = args.work_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. 保存 vertebra_distance_errors (字典) 为 CSV (长格式) - 保持不变
        errors_savename_csv = os.path.join(save_dir, f"vertebra_distance_errors_"+str(args.num_train_f_sample)+"_"+args.eval_data+"_"+str(args.num_code_run)+".csv")
        try:
            header = ['Vertebra', 'Error_Index', 'Distance_Error']
            with open(errors_savename_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for vertebra_label, error_list in vertebra_distance_errors.items():
                    for index, error_value in enumerate(error_list):
                        row = [vertebra_label, index, error_value]
                        writer.writerow(row)
            print(f"椎骨中心点误差数据 (CSV) 已保存到: {errors_savename_csv}")
        except Exception as e:
            print(f"保存椎骨误差数据 (CSV) 时出错: {e}")

        # 3. 保存 pos_list_pr (列表/数组) 为 CSV - 修改判断条件
        pr_pos_savename_csv = os.path.join(save_dir, f"pos_list_pr_"+str(args.num_train_f_sample)+"_"+args.eval_data+"_"+str(args.num_code_run)+".csv")
        try:
            header_pr = []
            # --- 修改在这里：使用 .size 检查 NumPy 数组是否为空 ---
            if isinstance(pos_list_pr, np.ndarray) and pos_list_pr.size > 0:
                # 假设 pos_list_pr 是一个 2D 数组 (每行是一个样本)
                # 获取列数（即每个样本的位置数量）
                num_cols = pos_list_pr.shape[1] if pos_list_pr.ndim == 2 else len(pos_list_pr[0])
                header_pr = [f'pr_pos_{i+1}' for i in range(num_cols)]
            elif isinstance(pos_list_pr, list) and pos_list_pr: # 如果还是列表，按原逻辑判断
                header_pr = [f'pr_pos_{i+1}' for i in range(len(pos_list_pr[0]))]
            # --- 修改结束 ---

            with open(pr_pos_savename_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                if header_pr:
                    writer.writerow(header_pr)
                # writerows 可以直接处理 NumPy 数组
                writer.writerows(pos_list_pr)
            print(f"预测Cobb角位置列表 (CSV) 已保存到: {pr_pos_savename_csv}")
        except Exception as e:
            print(f"保存 pos_list_pr (CSV) 时出错: {e}")

        # 4. 保存 pos_list_gt (列表/数组) 为 CSV - 修改判断条件 (同上)
        gt_pos_savename_csv = os.path.join(save_dir, f"pos_list_gt_"+str(args.num_train_f_sample)+"_"+args.eval_data+"_"+str(args.num_code_run)+".csv")
        try:
            header_gt = []
            # --- 修改在这里：使用 .size 检查 NumPy 数组是否为空 ---
            if isinstance(pos_list_gt, np.ndarray) and pos_list_gt.size > 0:
                num_cols = pos_list_gt.shape[1] if pos_list_gt.ndim == 2 else len(pos_list_gt[0])
                header_gt = [f'gt_pos_{i+1}' for i in range(num_cols)]
            elif isinstance(pos_list_gt, list) and pos_list_gt: # 如果还是列表，按原逻辑判断
                 header_gt = [f'gt_pos_{i+1}' for i in range(len(pos_list_gt[0]))]
            # --- 修改结束 ---

            with open(gt_pos_savename_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                if header_gt:
                    writer.writerow(header_gt)
                writer.writerows(pos_list_gt)
            print(f"真实Cobb角位置列表 (CSV) 已保存到: {gt_pos_savename_csv}")
        except Exception as e:
            print(f"保存 pos_list_gt (CSV) 时出错: {e}")

    def save_prediction_visualization(self, image_tensor, detections, gt_pts=None, img_id=""):
        # 1. 图像和颜色准备 (不变)
        img_for_vis = image_tensor[0].detach().cpu().numpy()
        img_for_vis = (img_for_vis + 0.5) * 255.0
        img_for_vis = np.clip(img_for_vis, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        vis_image = cv2.cvtColor(img_for_vis, cv2.COLOR_RGB2BGR).copy()
        colors = [tuple(c) for c in cv2.applyColorMap(np.arange(0, 255, 255//17, dtype=np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)]
        
        # 2. 绘制GT (不变)
        if gt_pts is not None:
            for pt in gt_pts.reshape(-1, 2):
                cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1)

        # 3. 准备小波重构参数 (不变)
        output_w = self.args.input_w // self.args.down_ratio
        normalization_factor = float(output_w)
        
        # 4. 遍历并绘制经过清理的预测结果
        for i, det in enumerate(detections):

            center_pred_feature_space = det[1:3]
            wavelet_coeffs_normalized = det[3:]
            
            # --- 【【【核心修改：集成轮廓清理逻辑】】】 ---
            
            # a. 反归一化小波系数，重建原始（可能有噪声的）轮廓
            wavelet_coeffs_denorm = wavelet_coeffs_normalized * normalization_factor
            noisy_contour = utils.reconstruct_contour_from_wavelet_np(
                wavelet_coeffs_denorm,
                center_pred_feature_space,
                self.args.wavelet_type,
                self.args.wavelet_level,
                self.args.coeffs_len_per_axis,
                self.args.num_dense_points
            )
            
            # b. 从噪声轮廓中拟合出规整的四个角点
            unordered_corners = utils.get_corners_from_contour(noisy_contour)
            fitted_corners = rearrange_pts(unordered_corners) # 假设 rearrange_pts 是您已有的函数
            
            # c. 从这四个干净的角点生成一个完美的、密集的矩形轮廓用于显示
            #    注意：这里我们直接使用角点来绘制四边形，而不是再生成密集点，这样更高效
            clean_contour_for_drawing = fitted_corners
            
            # --- 【【【修改结束】】】 ---
            
            # d. 缩放到原始图像尺寸并绘制清理后的轮廓
            contour_to_draw = (clean_contour_for_drawing * self.args.down_ratio).astype(np.int32)
            instance_color = [int(c) for c in colors[i]]
            
            # 使用 polylines 绘制四边形轮廓
            cv2.polylines(vis_image, [contour_to_draw], isClosed=True, color=instance_color, thickness=1)
            
            # 绘制中心点（不变）
            center_to_draw = (center_pred_feature_space * self.args.down_ratio).astype(np.int32)
            cv2.circle(vis_image, tuple(center_to_draw), 2, instance_color, -1)
            cv2.circle(vis_image, tuple(center_to_draw), 2, (0,0,0), 1)

        # 5. 保存图像
        output_path = os.path.join(self.img_show_dir, f"{img_id}.jpg")
        cv2.imwrite(output_path, vis_image)

    def save_prediction_visualization_ori(self, image_tensor, detections, gt_pts=None, img_id=""):
        # 1. 图像和颜色准备 (不变)
        img_for_vis = image_tensor[0].detach().cpu().numpy()
        img_for_vis = (img_for_vis + 0.5) * 255.0
        img_for_vis = np.clip(img_for_vis, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        vis_image = cv2.cvtColor(img_for_vis, cv2.COLOR_RGB2BGR).copy()
        colors = [tuple(c) for c in cv2.applyColorMap(np.arange(0, 255, 255//17, dtype=np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)]
        
        '''# 2. 绘制GT (不变)
        if gt_pts is not None:
            for pt in gt_pts.reshape(-1, 2):
                cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)'''

        # 3. 准备小波重构参数 (从 self.args 获取)
        output_w = self.args.input_w // self.args.down_ratio
        normalization_factor = float(output_w)
        
        # 4. 遍历并绘制预测结果
        for i, det in enumerate(detections):

            center_pred_feature_space = det[1:3]
            wavelet_coeffs_normalized = det[3:]
            
            # 反归一化小波系数
            wavelet_coeffs_denorm = wavelet_coeffs_normalized * normalization_factor
            
            # 使用新的NumPy重建函数
            reconstructed_contour = utils.reconstruct_contour_from_wavelet_np(
                wavelet_coeffs_denorm,
                center_pred_feature_space,
                self.args.wavelet_type,
                self.args.wavelet_level,
                self.args.coeffs_len_per_axis,
                self.args.num_dense_points
            )
            
            # 缩放到原始图像尺寸并绘制
            contour_to_draw = (reconstructed_contour * self.args.down_ratio).astype(np.int32)
            instance_color = [int(c) for c in colors[i]]
            cv2.polylines(vis_image, [contour_to_draw], isClosed=True, color=instance_color, thickness=1)
            
            center_to_draw = (center_pred_feature_space * self.args.down_ratio).astype(np.int32)
            cv2.circle(vis_image, tuple(center_to_draw), 2, instance_color, -1)
            cv2.circle(vis_image, tuple(center_to_draw), 2, (0,0,0), 1)

        # 5. 保存图像
        output_path = os.path.join(self.img_show_dir, f"{img_id}_ori.jpg")
        cv2.imwrite(output_path, vis_image)

    def get_spine_centers(self, pts_2, num_vertebrae=17):
        """
        从pts_2中解析脊椎中心点坐标
        
        Args:
            pts_2: 关键点坐标数组，形状为 (17 * 4, 2) 或 (17, 2)
            num_vertebrae: 脊椎数量，默认为17
        
        Returns:
            np.ndarray: 中心点坐标数组，形状为 (num_vertebrae, 2)
        """
        if pts_2.shape[0] == num_vertebrae * 4:
            # 处理角点模式：每4个角点计算一个中心
            centers = []
            for k in range(num_vertebrae):
                corners = pts_2[4*k : 4*k+4]
                center = np.mean(corners, axis=0)
                centers.append(center)
            return np.array(centers)
        elif pts_2.shape[0] == num_vertebrae:
            # 直接是中心点模式
            return pts_2.copy()
        else:
            raise ValueError(f"无效的pts_2形状: {pts_2.shape}")
        
    def save_heatmap(self, hm_tensor, output_path="heatmap.png"):
        """
        保存热图张量为彩色图像文件
        参数:
            hm_tensor (Tensor): 形状为 (1, 1, H, W) 的CUDA张量
            output_path (str): 输出文件路径
        """
        # 1. 转换张量到CPU并提取数据
        #hm_tensor = self._nms(hm_tensor)
        hm_np = hm_tensor.detach().cpu().numpy()[0,0]  # 去除批次和通道维度
        
        # 2. 对数变换增强低值区域可见性
        eps = 1e-12  # 防止log(0)
        log_hm = np.log(hm_np + eps)
        
        # 3. 特殊归一化处理（针对极小数优化）
        min_val = np.min(log_hm)
        max_val = np.max(log_hm)
        normalized = (log_hm - min_val) / (max_val - min_val + 1e-8)
        uint8_hm = (normalized * 255).astype(np.uint8)
        
        # 4. 应用JET颜色映射
        color_hm = cv2.applyColorMap(uint8_hm, cv2.COLORMAP_JET)
        
        # 5. 保存结果
        cv2.imwrite(output_path, color_hm)
        #print(f"Heatmap saved to {output_path}")

    def SMAPE_single_angle(self, gt_cobb_angles, pr_cobb_angles):
        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = out_abs
        term2 = out_add

        term2[term2==0] += 1e-5

        SMAPE = np.mean(term1 / term2 * 100)
        return SMAPE


