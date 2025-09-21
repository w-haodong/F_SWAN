import torch
import torch.nn as nn

from .dec_net import DecNet
from .DANN import GradientReversalLayer, DomainClassifier

class SCDNet(nn.Module):
    def __init__(self, peft_encoder, heads, vit_input_layer_indices):
        super(SCDNet, self).__init__()
        
        self.encoder = peft_encoder 
        self.vit_input_layer_indices = vit_input_layer_indices
        self.dec_net = DecNet(heads, vit_input_layer_indices=self.vit_input_layer_indices)

        # 【修改】初始化GRL时不再需要传入lambda_
        content_feature_dim = self.dec_net.final_feature_channels
        self.grl = GradientReversalLayer()
        self.domain_classifier = DomainClassifier(input_features=content_feature_dim)

    def forward(self, x, alpha=None):
        # 1. 编码器提取多尺度特征
        x_list = self.encoder(x, self.vit_input_layer_indices)
        
        # 2. 解码器进行特征解耦和任务预测
        dec_dict, content_features, style_features = self.dec_net(x_list)
        
        # 3. 对内容特征进行域对抗
        domain_pred = None # 默认值为None
        if alpha is not None:
            # 【修改】直接将alpha作为lambda_参数传入GRL的forward方法
            reversed_content_features = self.grl(content_features, alpha)
            domain_pred = self.domain_classifier(reversed_content_features)
        
        # 4. 返回所有需要用于计算损失的张量
        return dec_dict, domain_pred, content_features, style_features