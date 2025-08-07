import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

# see https://github.com/charlesCXK/TorchSemiSeg/blob/main/furnace/seg_opr/loss_opr.py

# ProbOhemCrossEntropy2d是个用于二维分割任务的损失函数
# 实现了一种概率版本的在线难例挖掘（OHEM）交叉熵损失函数，用于训练过程中的难例筛选
# 这个损失函数特别适用于需要处理大量负样本或类别不平衡的场景，例如语义分割任务。通过 OHEM，模型可以更加关注那些难以正确分类的样本
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index, reduction='mean', thresh=0.7, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()  # 调用父类的构造函数init
        self.ignore_index = ignore_index  # 忽略索引，用于指定在损失计算中应忽略的类别索引
        self.thresh = float(thresh)  # 损失函数的阈值参数，用于 OHEM 中的难例挖掘
        self.min_kept = int(min_kept)  # 保留的最小正样本数量
        self.down_ratio = down_ratio  # 下采样比例，用于调整损失函数的计算尺度
        if use_weight:  # 如果使用类别权重
            weight = torch.FloatTensor([1.0, 10.0]) # 定义每个类别的权重
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_index)
            # self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, weight=torch.tensor([1.0, 10.0]).cuda(int(os.environ["LOCAL_RANK"]),ignore_index=ignore_index)

        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_index)

    # pred 是模型的预测输出，target 是实际的标签
    def forward(self, pred, target):
        b, c, h, w = pred.size()  # 获取预测的批次大小、通道数、高度和宽度
        target = target.view(-1)  # 将标签展平为一维
        valid_mask = target.ne(self.ignore_index)  # 有效的掩码，排除忽略索引
        target = target * valid_mask.long()  # 应用有效掩码
        num_valid = valid_mask.sum()  # 计算有效像素的数量

        prob = F.softmax(pred, dim=1)  # 计算预测的 softmax 概率
        prob = (prob.transpose(0, 1)).reshape(c, -1)  # 调整概率的形状并应用有效掩码

        # OHEM 逻辑，根据阈值和最小保留数量调整目标掩码
        # 根据预测概率和预设的阈值，保留模型预测错误或者预测置信度低的样本，使模型在训练过程中更加关注这些难例。
        if self.min_kept > num_valid:
            pass  # 如果有效像素少于 min_kept，则不进行 OHEM
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)  # 将无效像素的概率设置为1，后续计算中忽略
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]  # 获取每个像素对应目标类别的概率
            threshold = self.thresh  # 根据阈值和最小保留数量确定难例
            if self.min_kept > 0:  # 找到最小概率的索引，并设置阈值
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                # 筛选难例
                kept_mask = mask_prob.le(threshold)  # 创建一个掩码，保留概率低于阈值的像素
                target = target * kept_mask.long()  # 更新目标值和有效掩码，只保留难例像素
                valid_mask = valid_mask * kept_mask

        # 应用最终的有效掩码到目标上，并将其恢复为原始形状
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        # 计算损失，使用配置的交叉熵损失函数
        return self.criterion(pred, target)
