import argparse
from operation import train
from operation import test
from operation import eval
import numpy as np
import pywt
import os
join = os.path.join
from models.sam.segment_anything import build_sam_encoder_only
# +++ PEFT/LoRA 导入 +++
from peft import get_peft_model, LoraConfig
import torchvision
torchvision.disable_beta_transforms_warning()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='w-haodong Test For Socliosis with F-SWAN - 2025 09')
    # 基础参数
    parser.add_argument('--device', type=str, default='cuda:1', help='GPU')
    parser.add_argument('--work_dir', type=str, default='/CSTemp/whd/work/sco_syb/whd_gitHub/', help='work directory')
    parser.add_argument('--num_epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=1024, help='input height')
    parser.add_argument('--input_w', type=int, default=1024, help='input width')
    parser.add_argument('--K', type=int, default=17, help='maximum of objects')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--resume', type=str, default='model_10_200.pth', help='weights_spinal to be resumed')
    parser.add_argument('--num_train_f_sample', type=int, default=10) #少样本训练数量
    parser.add_argument('--num_code_run', type=int, default=1, help='Number of code running')
    parser.add_argument('--phase', type=str, default='test', help='train, test, or eval')
    parser.add_argument('--eval_data', type=str, default='test')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='batch_size * accumulation_steps')
    # SAM par
    parser.add_argument("-chk", "--checkpoint", type=str, default="/CSTemp/whd/work/sco_syb/whd_v4/models/sam/work_dir/sam_vit_b_01ec64.pth", help="path to the trained model")
    parser.add_argument('--vit_input_layer_indices', type=int, default=[0,2,5], help='input with medsam for decnet')
    parser.add_argument('--img_dir', type=str, default='/CSTemp/whd/datasets/AASCE/bm_data/new_bm_data_', help="Root directory of the dataset, which contains 'train' and 'val' subfolders.")
    parser.add_argument('--ssda_mode', type=str, default='joint_train', choices=['none', 'joint_train'], help="Enable Unsupervised Domain Adaptation mode. 'none' for standard supervised training, 'joint_train' for our new strategy.")
    parser.add_argument('--warmup_epochs', type=int, default=1, help="Number of epochs to pre-train on source domain only (supervised warm-up).")
    parser.add_argument('--consistency_weight_max', type=float, default=0.1, help="The maximum weight for the unsupervised consistency loss.")
    parser.add_argument('--consistency_start_epochs', type=int, default=40, help="Number of epochs over which the consistency weight start.")
    parser.add_argument('--ema_start_epoch', type=int, default=35, 
                        help="The epoch at which the EMA teacher is reset to the student's state and begins updating.")
    parser.add_argument('--consistency_rampup_epochs', type=int, default=20, help="Number of epochs over which the consistency weight ramps up from 0 to its max value.")
    parser.add_argument('--uda_top_k_schedule', type=int, default=[10, 17, 10], help="Defines the Top-K curriculum. E.g., '--uda_topk_schedule 5 17 50'. Default is '5 17 50'.")
    parser.add_argument('--pseudo_label_thresh', type=float, default=0.2, help="Confidence threshold for filtering pseudo labels. Only heatmaps with max value > threshold are used.")
    # EMA Teacher 的衰減率/動量參數
    parser.add_argument('--ema_decay', type=float, default=0.995, 
                        help="Decay rate for the EMA teacher model. A high value close to 1 means slow updates.")
    # 小波变换描述轮廓配置
    parser.add_argument('--wavelet_type', type=str, default='db4', help="Type of wavelet basis, e.g., 'db4'. Must match dataset config.")
    parser.add_argument('--wavelet_level', type=int, default=4, help='Level of wavelet decomposition. Must match dataset config.')
    parser.add_argument('--num_dense_points', type=int, default=32, help='Number of points in the dense contour. Must match dataset config.')
    parser.add_argument('--vis_thresh', type=float, default=0.05, help='Confidence threshold for visualizing detections.')
    args = parser.parse_args()
    dummy_signal = np.zeros(args.num_dense_points)
    dummy_coeffs = pywt.wavedec(dummy_signal, args.wavelet_type, level=args.wavelet_level)
    coeffs_len_per_axis = sum(len(c) for c in dummy_coeffs)
    # 将计算出的正确值，作为新的属性添加到args对象中
    args.coeffs_len_per_axis = coeffs_len_per_axis
    args.data_dir = args.img_dir + str(args.num_train_f_sample)
    return args

if __name__ == '__main__':
    args = parse_args()
    device = args.device

    # 1. 正常加载原始SAM模型
    sam_model = build_sam_encoder_only(checkpoint_path=args.checkpoint)
    sam_model = sam_model.to(device)
    sam_model.eval()

    # 2. 定义LoRA配置
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.1, bias="none")

    # 3. 创建PEFT编码器
    print("Wrapping the image encoder with PEFT (LoRA)...")
    peft_encoder = get_peft_model(sam_model, lora_config)
    peft_encoder.print_trainable_parameters()

    if args.phase == 'test':
        is_object = test.Network(args, peft_encoder)
        is_object.test(args)
