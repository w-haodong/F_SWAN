import argparse
from operation import train
from operation import test
from operation import eval
import numpy as np
import pywt
import os
import sys
import torch
import torch.nn as nn

from models.sam.segment_anything import build_sam_encoder_only
from peft import get_peft_model, LoraConfig

# ====== 新增：THOP ======
try:
    from thop import profile
except Exception as e:
    profile = None

import torchvision
torchvision.disable_beta_transforms_warning()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---- 让 dinov3 作为顶层包可见（models/dino/dinov3/...）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DINO_ROOT = os.path.join(_THIS_DIR, 'models', 'dino')  # 该目录下应有 dinov3/
if _DINO_ROOT not in sys.path:
    sys.path.insert(0, _DINO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='w-haodong Test For Socliosis with big model - 2025 09')
    # 基础参数
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU')
    parser.add_argument('--work_dir', type=str, default='/CSTemp/whd/work/sco_syb/dino_syb/whd_v6_dino', help='work directory')
    parser.add_argument('--num_epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=1280, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=17, help='maximum of objects')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--resume', type=str, default='model_10_250.pth', help='weights_spinal to be resumed')
    parser.add_argument('--num_train_f_sample', type=int, default=10) # 少样本训练数量
    parser.add_argument('--num_code_run', type=int, default=0, help='Number of code running')

    # ====== 修改：加入 flops 选项 ======
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test', 'eval', 'flops'],
                        help='train, test, eval, or flops')
    parser.add_argument('--eval_data', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='spinal', help='dataset name')

    parser.add_argument('--accumulation_steps', type=int, default=4, help='batch_size * accumulation_steps')

    # ===== 编码器相关参数（encoder 选择 + DINO 权重） =====
    parser.add_argument(
        "-chk", "--checkpoint",
        type=str,
        default="./models/pth/sam_vit_b_01ec64.pth",
        help="path to the trained SAM model"
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='dino',
        choices=['sam', 'dino'],
        help='Backbone encoder type: sam or dino'
    )
    parser.add_argument(
        '--dino_checkpoint',
        type=str,
        default='./models/pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        help="Path to DINOv3 ViT-B/16 weights."
    )

    parser.add_argument('--vit_input_layer_indices', type=int, nargs='+', default=[0, 2, 5, 11],
                        help='which ViT layers to output for decoder')

    parser.add_argument('--img_dir', type=str,
                        default='/CSTemp/whd/datasets/AASCE/bm_data/new_bm_data_',
                        help="Root directory of the dataset, which contains 'train' and 'val' subfolders.")

    parser.add_argument('--ssda_mode', type=str, default='joint_train', choices=['none', 'joint_train'])
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--consistency_weight_max', type=float, default=1)
    parser.add_argument('--consistency_start_epochs', type=int, default=30)
    parser.add_argument('--ema_start_epoch', type=int, default=25)
    parser.add_argument('--consistency_rampup_epochs', type=int, default=30)
    parser.add_argument('--uda_top_k_schedule', type=int, nargs='+', default=[10, 17, 10])
    parser.add_argument('--pseudo_label_thresh', type=float, default=0.2)
    parser.add_argument('--ema_decay', type=float, default=0.995)

    # 小波配置
    parser.add_argument('--wavelet_type', type=str, default='db4')
    parser.add_argument('--wavelet_level', type=int, default=4)
    parser.add_argument('--num_dense_points', type=int, default=32)
    parser.add_argument('--vis_thresh', type=float, default=0.05)

    args = parser.parse_args()

    # ---- 小波系数长度计算（原样保留）----
    dummy_signal = np.zeros(args.num_dense_points)
    dummy_coeffs = pywt.wavedec(dummy_signal, args.wavelet_type, level=args.wavelet_level)
    coeffs_len_per_axis = sum(len(c) for c in dummy_coeffs)
    args.coeffs_len_per_axis = coeffs_len_per_axis

    args.data_dir = args.img_dir + str(args.num_train_f_sample)
    return args


def _build_encoder(args, device):
    if args.encoder == 'sam':
        print(f"[Encoder] Using SAM encoder (sam_vit_b).")
        print(f"[Encoder] Loading checkpoint: {args.checkpoint}")
        enc = build_sam_encoder_only(checkpoint_path=args.checkpoint)
        enc = enc.to(device)
        enc.eval()
        return enc

    print(f"[Encoder] Using DINOv3 encoder (dinov3_vitb16).")
    print(f"[Encoder] Loading checkpoint: {args.dino_checkpoint}")
    try:
        from dinov3.hub.backbones import dinov3_vitb16
    except Exception as e:
        raise ImportError(
            "Cannot import dinov3_vitb16 from dinov3.hub.backbones. "
            "请确认 models/dino/dinov3 目录结构正确，且本脚本顶部已将 models/dino 加入 sys.path。"
        ) from e

    dino_model = dinov3_vitb16(
        pretrained=True,
        weights=args.dino_checkpoint,
        check_hash=False,
    )
    dino_model = dino_model.to(device)
    dino_model.eval()
    return dino_model


# ====== 新增：THOP wrapper（避免 thop 处理 dict 输出不稳定）======
class _FlopsWrapper(nn.Module):
    def __init__(self, spinal_model: nn.Module):
        super().__init__()
        self.m = spinal_model

    def forward(self, x):
        # 你的 SpineNet forward: (dec_dict, domain_pred, content, style)
        dec_dict, _, _, _ = self.m(x, alpha=None)
        # 返回一个 tensor 给 thop（随便选一个头即可）
        return dec_dict['hm']


def _run_flops(args, peft_encoder):
    if profile is None:
        raise ImportError("thop 未安装：请先 `pip install thop`")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 你的 heads（与训练一致）
    total_wavelet_coeffs = args.coeffs_len_per_axis * 2
    heads = {
        'hm': args.num_classes,
        'reg': 2 * args.num_classes,
        'wavelet': total_wavelet_coeffs
    }

    from models import spinal_net
    model = spinal_net.SpineNet(
        peft_encoder=peft_encoder,
        heads=heads,
        vit_input_layer_indices=args.vit_input_layer_indices
    ).to(device).eval()

    wrapped = _FlopsWrapper(model).to(device).eval()

    # dummy 输入（与你实际输入分辨率一致）
    x = torch.randn(1, 3, args.input_h, args.input_w, device=device)

    with torch.no_grad():
        macs, params = profile(wrapped, inputs=(x,), verbose=False)

    # thop 输出通常更接近 MACs；如果你按 FLOPs=2*MACs，则：
    flops = 2.0 * macs

    print("\n========== Model Complexity (THOP) ==========")
    print(f"Backbone: {args.encoder.upper()} + LoRA | Layers: {args.vit_input_layer_indices}")
    print(f"Input: {tuple(x.shape)}")
    print(f"Params: {params / 1e6:.3f} M")
    print(f"MACs :  {macs / 1e9:.3f} G")
    print(f"FLOPs:  {flops / 1e9:.3f} G  (按 FLOPs=2*MACs 口径)")
    print("===========================================\n")


if __name__ == '__main__':
    args = parse_args()
    device = args.device

    # 1) 构建编码器（SAM 或 DINO）
    encoder = _build_encoder(args, device)

    # 2) LoRA（保持你原来的设置）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none"
    )

    print("Wrapping the image encoder with PEFT (LoRA)...")
    peft_encoder = get_peft_model(encoder, lora_config)
    peft_encoder.print_trainable_parameters()
    if args.phase == 'test':
        is_object = test.Network(args, peft_encoder)
        is_object.test(args)


