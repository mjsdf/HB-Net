from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
# from util.patch import ContaminatedContrast
from util.patch import SemanticContrast
import torch.nn.functional as F


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        target_size = [x2.size(2), x2.size(3)]
        x1 = nn.functional.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        output = self.out_conv(x)
        return output, x  # 返回最终输出和最后一层特征


class ProjectionHead(nn.Module):
    """用于对比学习的投影头"""

    def __init__(self, in_dim, hidden_dim=256, out_dim=128, use_bn=True):
        super().__init__()
        if use_bn:
            self.projection = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
            )
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
            )

    def forward(self, x):
        x = self.projection(x)  # 投影到特征空间
        return F.normalize(x, dim=1)


class FeatureFusion(nn.Module):
    """编码器x3层与解码器输出层融合模块"""

    def __init__(self, encoder_dim, decoder_dim, out_dim):
        super().__init__()
        # 编码器x3层投影（128→out_dim）
        self.enc_proj = nn.Conv2d(encoder_dim, out_dim, kernel_size=1)
        # 解码器输出层投影（16→out_dim）
        self.dec_proj = nn.Conv2d(decoder_dim, out_dim, kernel_size=1)
        # 融合后的非线性变换
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        # 上采样编码器x3层至解码器尺寸（180x180）
        enc_feat_up = nn.functional.interpolate(enc_feat, size=dec_feat.shape[2:], mode='bilinear', align_corners=True)
        # 投影到相同通道数
        enc_proj = self.enc_proj(enc_feat_up)
        dec_proj = self.dec_proj(dec_feat)
        # 通道拼接与融合
        fused = torch.cat([enc_proj, dec_proj], dim=1)
        return self.fusion_conv(fused)


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.return_attn = False

    def enable_projection(self):
        "启用对比学习模块"
        if not hasattr(self, 'projection_head'):  # 防止重复初始化

           # 投影头
           self.projection_head = ProjectionHead(
               in_dim=16,  # 融合特征的维度
               hidden_dim=256,  # 若计算资源有限，可将hidden_dim从 256 降至 128，减少参数规模
               out_dim=128  # 对比学习的特征维度
           )
           # 将 contrast_loss 移动到 GPU 上
           if torch.cuda.is_available():
               self.projection_head.cuda()
           self.contrast_loss = SemanticContrast(temp=0.7)

        self.return_attn = True

    def forward(self, x, need_fp=False, return_attn=False):
        """修改前向传播以支持特征返回
        Args:
        return_features: 控制是否返回中间特征
        """
        feature = self.encoder(x)  # [x0, x1, x2, x3, x4]

        if need_fp:
            outs, decoder_feat = self.decoder([torch.cat((feat, nn.Dropout2d(0.5)(feat))) for feat in feature])
        else:
            outs, decoder_feat = self.decoder(feature)

        if return_attn:

            # 编码器第一层特征+尺寸适应
            features = F.interpolate(feature[0], size=x.shape[2:], mode='bilinear', align_corners=True)  # 如果用编码器x4特征对比学习，那么需要尺寸一致
            proj_feature = self.projection_head(features)
            return outs, proj_feature
        else:
            return outs

    def get_contrast_loss(self, features, images, labels):
        """对外暴露的对比损失计算接口"""
        return self.contrast_loss(features, images, labels)
