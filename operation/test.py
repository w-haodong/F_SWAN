import torch
import numpy as np
from models import scd_net
import torch.nn.functional as F
import cv2
from operation import decoder
import os
from datasets.dataset import BaseDataset
from datasets import draw_points
import matplotlib
from operation import decoder, cobb_evaluate_base, utils
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import scipy.io as sio


def tensor_to_bgr_uint8(image_tensor_1chw: torch.Tensor) -> np.ndarray:
    """
    image_tensor_1chw: (1,3,H,W), normalized by /255 - 0.5
    returns: uint8 BGR image (H,W,3)
    """
    img = image_tensor_1chw[0].detach().cpu().numpy()  # (3,H,W)
    img = (img + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)  # (H,W,3) BGR
    return img

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args, peft_encoder):
        torch.manual_seed(317)
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.total_wavelet_coeffs = self.args.coeffs_len_per_axis * 2
        heads = {'hm': self.args.num_classes,
                 'reg': 2 * self.args.num_classes,
                 'wavelet': self.total_wavelet_coeffs}

        self.model = scd_net.SCDNet(peft_encoder=peft_encoder,
                                    heads=heads,
                                    vit_input_layer_indices=args.vit_input_layer_indices
                                    )
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder()
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights_spinal from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def test(self, args):
        save_path = args.work_dir + '/weights_spinal'
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dsets = self.dataset['spinal'](args=args, phase='test', eval_data=args.eval_data)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
        img_show_dir = args.work_dir + '/img_show'
        if not os.path.exists(img_show_dir):
            os.makedirs(img_show_dir)
            print(f"创建目录用于保存 .jpg 文件: {img_show_dir}")

        for cnt, data_dict in enumerate(data_loader):
            image_tensor = data_dict['input'].to(self.device)
            img_id = data_dict['img_id'][0]

            # =========================
            input_bgr = tensor_to_bgr_uint8(data_dict['input'])  # (H,W,3) BGR，H/W= input_h/input_w
            image_cobb_pr = input_bgr.copy()
            h_orig, w_orig = image_cobb_pr.shape[:2]

            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))

            with torch.no_grad():
                pr_decs, _, _, _ = self.model(image_tensor)
                # 解码器总是返回17个候选椎体
                hm = pr_decs['hm']
                reg = pr_decs['reg']
                wavelet = pr_decs['wavelet']
                detections = self.decoder.decode(
                    heat=hm, reg=reg, wavelet=wavelet
                ).cpu().numpy()[0]

            torch.cuda.synchronize(self.device)

            all_pred_corners = []
            detections = sorted(detections, key=lambda d: d[2])  # 按y坐标排序
            for det in detections:
                contour_feat = utils.reconstruct_contour_from_wavelet_np(
                    det[3:] * (args.input_w / args.down_ratio),
                    det[1:3],
                    args.wavelet_type,
                    args.wavelet_level,
                    args.coeffs_len_per_axis,
                    args.num_dense_points)
                contour_img = contour_feat * args.down_ratio
                corners = utils.get_corners_from_contour(contour_img)
                all_pred_corners.append(corners)

            # 现在 all_pred_corners 列表里总是包含17个椎体的角点
            pr_landmarks = np.concatenate(all_pred_corners, axis=0)

            scale_w, scale_h = self.args.input_w / w_orig, self.args.input_h / h_orig
            pr_landmarks[:, 0] /= scale_w
            pr_landmarks[:, 1] /= scale_h

            # 计算预测Cobb角（坐标系与 image_cobb_pr 一致）
            pr_ca1, pr_ca2, pr_ca3, _ = cobb_evaluate_base.cobb_angle_calc(
                pr_landmarks, image_cobb_pr, is_train=False
            )
            print('opr_ca1: {:.2f}, opr_ca2: {:.2f}, opr_ca3: {:.2f}'.format(pr_ca1, pr_ca2, pr_ca3))

            cv2.imwrite(os.path.join(img_show_dir, 'ca_pr_' + img_id), image_cobb_pr)

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep
