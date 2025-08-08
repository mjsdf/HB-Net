import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


# 随机裁剪图像和验码到指定大小
def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0  # 计算宽度方向上的填充
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    # 随机选择一个区域进行裁剪
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


# 水平翻转图像和掩码
def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


# 标准化图像，将掩码转换为张量
def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


# 随机缩放图像和掩码
def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


# 对图像应用随机模糊
def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


# 生成cutmix数据增强的掩码
def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size)  # 初始化一个全0的掩码，大小与输入图像相同

    if random.random() > p:  # p为CutMix操作的概率，如果随机数大于p，则不执行CutMix
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size  # 计算混合区域的大小，np.random.uniform(size_min, size_max) 会生成一个在size_min和size_max之间均匀分布的随机浮点数
    # 循环直到生成一个在图像范围内的混合区域
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)  # 随机选择混合区域的宽高比

        cutmix_w = int(np.sqrt(size / ratio))  # 根据比例和面积计算混合区域的宽度和高度
        cutmix_h = int(np.sqrt(size * ratio))

        x = np.random.randint(0, img_size)  # 随机选择混合区域的起始坐标
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:  # 确保混合区域在图像范围内
            break

    # 创建cutmix混合区域的掩码
    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
