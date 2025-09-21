import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, List, Tuple, Union


class LayerNorm2d(nn.LayerNorm):
    """ 2D Layer Normalization. """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)

class FeatureProcessingBlock(nn.Module):
    """ 独立处理来自单个ViT块的特征 """
    def __init__(self, in_channels: int, out_channels: int,
                 norm_layer: Type[nn.Module] = LayerNorm2d,
                 act_layer: Type[nn.Module] = nn.SiLU):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)

class TopDownFusionBlock(nn.Module):
    """ 融合来自更高层级的特征（自顶向下）和侧向连接的特征 """
    def __init__(self, top_down_channels: int, lateral_channels: int, out_channels: int,
                 norm_layer: Type[nn.Module] = LayerNorm2d,
                 act_layer: Type[nn.Module] = nn.SiLU):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(top_down_channels + lateral_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(),
        )
    def forward(self, top_down_feat: torch.Tensor, lateral_feat: torch.Tensor) -> torch.Tensor:
        # 注意：在融合前，确保 top_down_feat 和 lateral_feat 具有相同的空间维度
        # 如果需要，可以在这里添加上采样或下采样
        if top_down_feat.shape[2:] != lateral_feat.shape[2:]:
            top_down_feat = F.interpolate(
                top_down_feat, size=lateral_feat.shape[2:], mode='bilinear', align_corners=False
            )
        x = torch.cat([top_down_feat, lateral_feat], dim=1)
        return self.fusion_conv(x)


# -----------------------------------------------------------------------------
# 核心解码器网络 (Core Decoder Network)
# -----------------------------------------------------------------------------

class DecNet(nn.Module):
    def __init__(self,
                 heads: Dict[str, int],
                 vit_input_layer_indices: Union[List[int], Tuple[int, ...]],
                 in_channels_per_block: int = 768,
                 processed_block_channels: int = 256,
                 top_down_fused_channels: int = 256,
                 final_feature_channels_for_heads: int = 64,
                 act_layer: Type[nn.Module] = nn.SiLU,
                 norm_layer: Type[nn.Module] = LayerNorm2d):
        super(DecNet, self).__init__()

        self.num_vit_block_inputs = len(vit_input_layer_indices)
        if self.num_vit_block_inputs < 1:
            raise ValueError("vit_input_layer_indices must provide at least one index.")

        self.heads = heads
        self.final_feature_channels = final_feature_channels_for_heads

        # 特征处理器 (Feature Processors)
        self.feature_processors = nn.ModuleList([
            FeatureProcessingBlock(in_channels_per_block, processed_block_channels,
                                   norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(self.num_vit_block_inputs)
        ])
        
        # 自顶向下融合路径 (Top-Down Fusion Path)
        self.top_down_fusion_modules = nn.ModuleList([
            TopDownFusionBlock(top_down_fused_channels, processed_block_channels, top_down_fused_channels,
                               norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(self.num_vit_block_inputs - 1)
        ])
        self.initial_top_path_adapter = nn.Sequential(
            nn.Conv2d(processed_block_channels, top_down_fused_channels, kernel_size=1, bias=False),
            norm_layer(top_down_fused_channels),
            act_layer()
        ) if processed_block_channels != top_down_fused_channels else nn.Identity()
        
        self.content_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(top_down_fused_channels, top_down_fused_channels // 2, kernel_size=3, padding=1, bias=False),
            norm_layer(top_down_fused_channels // 2),
            act_layer(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(top_down_fused_channels // 2, final_feature_channels_for_heads, kernel_size=3, padding=1, bias=False),
            norm_layer(final_feature_channels_for_heads),
            act_layer()
        )

        # 预测头 (Prediction Heads / Task Heads)
        conv_dim = 256
        final_kernel = 1
        for head_name in self.heads:
            classes = self.heads[head_name]
            if head_name == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(self.final_feature_channels, conv_dim, kernel_size=3, padding=1, bias=True),
                    norm_layer(conv_dim), 
                    act_layer(),
                    nn.Conv2d(conv_dim, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                # 初始化热图头的偏置，使其在训练初期倾向于输出较低的值
                fc[-1].bias.data.fill_(-2.19)
            else: # for reg and wavelet heads
                fc = nn.Sequential(
                    nn.Conv2d(self.final_feature_channels, conv_dim, kernel_size=3, padding=1, bias=True),
                    norm_layer(conv_dim), 
                    act_layer(),
                    nn.Conv2d(conv_dim, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                self._fill_fc_weights(fc)
            self.__setattr__(head_name, fc)

        # 创建独立的 'style_path'
        style_feature_channels = final_feature_channels_for_heads // 2 
        self.style_path = nn.Sequential(
            nn.Conv2d(top_down_fused_channels, top_down_fused_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(top_down_fused_channels),
            nn.ReLU(inplace=True),
            # 使用1x1卷积高效降维
            nn.Conv2d(top_down_fused_channels, style_feature_channels, kernel_size=1, bias=True),
            norm_layer(style_feature_channels),
            nn.ReLU(inplace=True)
        )
        self._fill_fc_weights(self.style_path)

    def _fill_fc_weights(self, layers):
        """ 初始化权重 """
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, vit_block_outputs: List[torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # FPN风格的自顶向下融合
        if not isinstance(vit_block_outputs, list) or len(vit_block_outputs) != self.num_vit_block_inputs:
            raise ValueError(f"Expected {self.num_vit_block_inputs} features, got {len(vit_block_outputs)}")
        
        # 首先独立处理来自ViT不同层的特征
        processed_P = [self.feature_processors[i](vit_block_outputs[i]) for i in range(self.num_vit_block_inputs)]
        
        # 初始化最顶层的特征
        current_fused_feature = self.initial_top_path_adapter(processed_P[-1])
        
        # 从倒数第二层开始，自顶向下迭代融合
        for i in range(self.num_vit_block_inputs - 2, -1, -1):
            lateral_feature_P_i = processed_P[i]
            # 计算正确的融合模块索引
            fusion_module_idx = (self.num_vit_block_inputs - 2) - i
            current_fused_feature = self.top_down_fusion_modules[fusion_module_idx](current_fused_feature, lateral_feature_P_i)
        
        
        # 内容路径: 生成用于任务的特征图
        content_features_map = self.content_path(current_fused_feature)
        
        # 任务预测路径 (使用内容特征图)
        dec_dict = {}
        if hasattr(self, 'hm'):
            hm_output = self.hm(content_features_map)
            dec_dict['hm'] = torch.sigmoid(hm_output) # 保持sigmoid激活
        if hasattr(self, 'reg'):
            dec_dict['reg'] = self.reg(content_features_map)
        if hasattr(self, 'wavelet'):
            dec_dict['wavelet'] = self.wavelet(content_features_map)
            
        # 内容特征池化 (用于损失计算)
        content_features_pooled = F.adaptive_avg_pool2d(content_features_map, 1).view(content_features_map.size(0), -1)

        # 风格路径: 生成风格特征 (注意输入是 current_fused_feature，保证路径独立)
        style_features_map = self.style_path(current_fused_feature)
        style_features_pooled = F.adaptive_avg_pool2d(style_features_map, 1).view(style_features_map.size(0), -1)
        
        # 返回三组输出: 任务预测字典, 池化后的内容特征, 池化后的风格特征
        return dec_dict, content_features_pooled, style_features_pooled