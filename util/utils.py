import numpy as np
import logging
import os
import cv2


# 计算模型参数量的函数
def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


# 根据数据集生成颜色映射
def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')  # 初始化一个 256x3 的零数组 cmap，用于存储颜色值，数据类型为无符号 8 位整数

    # 如果数据集是 'pascal' 或 'coco' ，通过位操作生成 256 种颜色值并存储在 cmap 中
    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255, 0, 0])
        cmap[13] = np.array([0, 0, 142])
        cmap[14] = np.array([0, 0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0, 0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'liefeng':
        cmap[0] = np.array([0, 0, 0])  # 背景颜色，黑色
        cmap[1] = np.array([1, 1, 1])  # 孔缝颜色

    return cmap


#用于计算和存储一系列数值的平均值和当前值，有两种工作模式，取决于初始化时给定的 length 参数。
#如果 length 大于 0 ，它会使用一个固定长度的列表来保存历史值，并根据这个列表计算平均值。每次更新值时，如果列表长度超过了 length ，会删除最早的元素。
#如果 length 为 0 ，则通过累计数值的总和以及数量来计算平均值。
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        """
        参数:
        - length (int): 用于控制历史值存储的长度。如果 length > 0，则使用固定长度的历史值列表；否则，使用总和与计数来计算平均值
        """
        self.length = length
        self.reset()

    def reset(self):
        """
        重置平均计算器
        如果 length > 0，则清空历史值列表；否则，将计数、总和、当前值和平均值重置为初始状态
        """
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        """
        更新平均值和当前值
        参数:
        - val (float): 要添加的值
        - num (int): 值的数量（默认为 1）
        如果使用历史值列表（length > 0），则添加新值并处理长度限制。否则，更新总和与计数，并计算新的平均值
        """
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


# 计算交集和并集（原代码）
def intersectionAndUnion1(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# 2.修改的加权IOU，仅对裂缝类别进行加权计算
def generate_weight_map(gt_mask, kernel_size=5, high_weight=1.0, low_weight=0.2):
    """
    生成权重图，支持二维和三维的 gt_mask 输入。
    若 gt_mask 为二维，直接处理；若为三维，则逐张处理每个样本。
    """
    # 检查 gt_mask 是否为三维
    if gt_mask.ndim == 3:
        # 初始化存储权重图的列表
        weight_maps = []
        # 遍历批次中的每个样本
        for single_gt_mask in gt_mask:
            # 标注区域直接赋予高权重
            weight_map = np.ones_like(single_gt_mask, dtype=np.float32) * low_weight
            weight_map[single_gt_mask > 0] = high_weight

            # 对标注区域边缘进行膨胀，生成过渡权重区
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # 将 single_gt_mask 转换为 uint8 类型
            single_gt_mask_uint8 = single_gt_mask.astype(np.uint8)
            dilated = cv2.dilate(single_gt_mask_uint8, kernel, iterations=1)
            transition_region = dilated - single_gt_mask_uint8
            weight_map[transition_region > 0] = (high_weight + low_weight) / 2

            # 将单张图像的权重图添加到列表中
            weight_maps.append(weight_map)
        # 将列表中的权重图堆叠成一个三维数组
        return np.stack(weight_maps, axis=0)
    elif gt_mask.ndim == 2:
        # 若输入为二维，直接处理
        weight_map = np.ones_like(gt_mask, dtype=np.float32) * low_weight
        weight_map[gt_mask > 0] = high_weight

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # 将 gt_mask 转换为 uint8 类型
        gt_mask_uint8 = gt_mask.astype(np.uint8)
        dilated = cv2.dilate(gt_mask_uint8, kernel, iterations=1)
        transition_region = dilated - gt_mask_uint8
        weight_map[transition_region > 0] = (high_weight + low_weight) / 2

        return weight_map
    else:
        # 若输入维度既不是二维也不是三维，抛出错误
        raise ValueError("Input gt_mask should be either 2D or 3D array.")
def intersectionAndUnion(output, target, K, ignore_index=1000):
    weight_map = generate_weight_map(target)

    assert output.ndim in [1, 2, 3] # 确保输出的维度在 1、2 或 3 维
    # 确保输出、目标和权重图的形状相同
    assert output.shape == target.shape
    assert output.shape == weight_map.shape

    # 将输出、目标和权重图重塑为一维
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    weight_map = weight_map.reshape(weight_map.size)

    # 将目标中等于忽略索引的值，在输出中也设置为忽略索引
    output[np.where(target == ignore_index)[0]] = ignore_index

    # 初始化普通交集、并集和加权交集、并集、加权目标
    intersection = np.zeros(K)
    union = np.zeros(K)

    weighted_intersection = np.zeros(K)
    weighted_union = np.zeros(K)
    weighted_target = np.zeros(K)
    for i in range(K):
        if i == 0:  # 背景类别采用普通 IOU 计算
            output_i_indices = np.where(output == i)[0]
            target_i_indices = np.where(target == i)[0]

            intersection_i_indices = np.intersect1d(output_i_indices, target_i_indices)
            intersection[i] = len(intersection_i_indices)
            union[i] = len(np.union1d(output_i_indices, target_i_indices))
        else:  # 裂缝类别采用加权 IOU 计算
            output_i_indices = np.where(output == i)[0]
            target_i_indices = np.where(target == i)[0]

            # 计算当前类别的加权交集
            intersection_i_indices = np.intersect1d(output_i_indices, target_i_indices)
            weighted_intersection[i] = np.sum(weight_map[intersection_i_indices])

            # 计算当前类别的加权输出
            weighted_output = np.sum(weight_map[output_i_indices])
            # 计算当前类别的加权目标
            weighted_target[i] = np.sum(weight_map[target_i_indices])

            # 计算加权并集
            weighted_union[i] = weighted_output + weighted_target[i] - weighted_intersection[i]

        # 合并结果
    final_intersection = np.where(np.arange(K) == 0, intersection, weighted_intersection)
    final_union = np.where(np.arange(K) == 0, union, weighted_union)

    return final_intersection, final_union, weighted_target

logs = set()


# 初始化日志
def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
