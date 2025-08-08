from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        # 初始化函数，设置数据集的基本信息
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:  # 使用open函数以读模式打开id_path指定的文件
                self.ids = f.read().splitlines()  # 读取文件内容，splitlines()方法将文件按行分割成列表，并去除每行末尾的换行符
            # 如果是训练有标签的数据，并且指定了nsample，则调整ids列表
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))  # 计算至少需要多少份ids列表来满足nsample
                self.ids = self.ids[:nsample]  # 然后将ids列表重复相应份数，再取前nsample个元素，以确保有足够数量的样本
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:  # 将变量 name 的值插入到 %s 的位置，从而得到具体的文件路径
                self.ids = f.read().splitlines()

    # 根据索引item获取和处理一个样本
    # 对不同的模式（验证模式、有标签训练模式、无标签训练模式）进行不同的处理
    # 验证模式：对图像和掩码进行标准化处理后返回
    # 有标签训练模式：对图像和掩码进行一系列处理后标准化并返回
    # 无标签训练模式：创建图像的多个变体，进行不同的数据增强操作，处理掩码和忽略掩码，并最终返回处理后的图像和相关信息
    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')  # 打开图像并转换为RGB模式
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))  # 加载掩码

        # 验证模式
        if self.mode == 'val':
            img, mask = normalize(img, mask)  # 标准化图像和掩码
            return img, mask, id

        # 训练模式下的图像和掩码处理
        # img, mask = resize(img, mask, (0.5, 2.0))  # 调整图像和掩码的大小
        ignore_value = 254 if self.mode == 'train_u' else 255  # 根据训练模式设置忽略值
        img, mask = crop(img, mask, self.size, ignore_value)  # 裁剪图像和掩码
        img, mask = hflip(img, mask, p=0.5)  # 以 0.5 的概率水平翻转

        # 有标签训练模式
        if self.mode == 'train_l':
            return normalize(img, mask)

        # 无标签训练模式，复制图像创建多个图像变体
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        # 对img_s1应用数据增强
        if random.random() < 0.8:  # 以0.8的概率执行以下增强
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)  # 颜色抖动
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)  # 以 0.2 的概率转换为灰度图
        img_s1 = blur(img_s1, p=0.5)  # 以 0.5 的概率进行模糊处理
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)  # 获取 CutMix 框

        # 对img_s2应用数据增强
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))  # 创建全零的忽略掩码

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)  # 标准化 img_s1 和忽略掩码
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()  # 将掩码转化为长整型张量，为了使掩码数据类型和形状适合深度学习模型处理
        ignore_mask[mask == 254] = 255  # 将某些特定的类别或区域标记为在训练过程中应该被忽略，不参与损失计算

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)  # 返回数据集中的样本数量
