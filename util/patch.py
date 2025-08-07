import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.filters.rank import entropy as skimage_entropy  # 对每个像素计算其邻域内的熵
from skimage.morphology import disk, binary_opening
import cv2


# 反标准化函数
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    img = img * std + mean
    return img.clamp(0, 1)


class SemanticContrast(nn.Module):
    def __init__(self, temp=0.7, entropy_weight=0.85, margin=0.3, patch_sizes=[105],
                 gray_threshold=(33, 150), entropy_threshold=(3.6, 6.0), crack_threshold=(10, 100), num_classes=2,
                 max_samples_per_class=3000, ohem_thresh=0.7, ohem_min_kept=3500,
                 use_dynamic_temp=True, temp_base=0.7, temp_scale=0.5,
                 temp_center=0.5):
        super().__init__()
        self.temp = temp  # 温度参数，用于计算相似度
        self.entropy_weight = entropy_weight  # 熵权重，用于加权特征，控制熵的影响强度
        self.margin = margin  # 边界裕度，防止过拟合，未使用。规定了不相似样本之间的特征距离必须大于这个值

        self.patch_sizes = patch_sizes
        self.gray_threshold = gray_threshold
        self.entropy_threshold = entropy_threshold
        self.crack_threshold = crack_threshold

        self.num_classes = num_classes  # 新增类别数参数
        self.max_samples = max_samples_per_class  # 控制最大样本数
        self.ohem_thresh = ohem_thresh  # OHEM 阈值
        self.ohem_min_kept = ohem_min_kept  # 最小保留样本数

        self.use_dynamic_temp = use_dynamic_temp  # 是否使用动态温度
        self.temp_base = temp_base  # 基础温度值
        self.temp_scale = temp_scale  # 温度缩放因子
        self.temp_center = temp_center  # 温度中心点

    def _apply_ohem(self, loss, weights, min_kept):  # 筛选困难样本，根据损失值排序并保留前min_kept个困难样本
        if loss.numel() == 0:
            return torch.tensor(0.0, device=loss.device)

        # 动态调整min_kept
        min_kept = min(min_kept, loss.numel())
        min_kept = max(min_kept, 1)  # 确保至少保留1个样本

        # 正样本选择损失值最大的（难以区分的样本）
        # 负样本选择损失值最大的（容易混淆的样本）
        sorted_loss, indices = torch.sort(loss, descending=True)

        threshold = sorted_loss[min_kept - 1]
        # 应用阈值筛选
        mask = loss >= threshold
        selected_loss = loss[mask]
        selected_weights = weights[mask]

        if selected_loss.numel() == 0:
            return torch.tensor(0.0, device=loss.device)

        return selected_loss.sum() / (selected_weights.sum() + 1e-8)

    def _precompute_entropy(self, img_gray):
        """批量预计算熵图（GPU优化）"""
        B, H, W = img_gray.shape
        entropies = torch.zeros((B, H, W), device=img_gray.device)

        for i in range(B):
            # 将torch.Tensor转换为numpy.ndarray
            img_gray_np = img_gray[i].cpu().numpy()
            ent = skimage_entropy(img_gray_np.astype(np.uint8), disk(2))
            entropies[i] = torch.from_numpy(ent).to(img_gray.device)
        return entropies  # [B,H,W]

    def _get_valid_patches(self, gray_img, label_img, full_entropy, device):
        """向量化获取有效补丁"""
        valid_patches = []

        for size in self.patch_sizes:
            # 生成所有可能的补丁坐标（显式指定indexing）
            y_coords = torch.arange(0, gray_img.shape[0] - size + 1, size, device=device)
            x_coords = torch.arange(0, gray_img.shape[1] - size + 1, size, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # 修复警告
            coords = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)

            for y, x in coords:
                y_end = y + size
                x_end = x + size
                patch = gray_img[y:y_end, x:x_end]

                # 灰度筛选
                patch_avg_gray = patch.float().mean()  # 计算图像块的平均灰度
                if not (self.gray_threshold[0] < patch_avg_gray < self.gray_threshold[1]):
                    continue

                # 熵筛选
                hist = torch.histc(patch.flatten().float(), bins=256, min=0, max=255)
                hist = (hist + 1e-8) / (hist.sum() + 256 * 1e-8)
                patch_entropy = -torch.sum(hist * torch.log(hist))
                if not (self.entropy_threshold[0] < patch_entropy < self.entropy_threshold[1]):
                    continue

                # 计算裂缝数量（如果提供了标签图像）
                crack_count = 0
                if label_img is not None:
                    label_patch = label_img[y:y_end, x:x_end]
                    # 确保标签补丁不为空
                    if label_patch.numel() > 0:
                        # 转换为NumPy数组（移到CPU并转为uint8）
                        label_patch = label_patch.cpu().numpy().astype(np.uint8)
                        # 连通性分析
                        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(label_patch, connectivity=8)
                        # 排除背景（标签为0）
                        crack_count = num_labels - 1
                if not (self.crack_threshold[0] < crack_count < self.crack_threshold[1]):
                    continue
                # 收集有效补丁
                valid_patches.append((y, x, size))

        return valid_patches

    def forward(self, feat, images, labels):
        """
        feat: 编码器或解码器的特征 [B,C,H,W]
        images: 原始图像 [B,3,H,W]
        labels: 标签图 [B,H,W]（假设0为背景，1为裂缝）
        """
        B, C, H, W = feat.shape
        device = feat.device

        # 反标准化
        images = denormalize(images)
        images = (images * 255).clamp(0, 255).to(torch.uint8)  # 转换为0-255的uint8

        # 批量灰度转换
        img_gray = torch.stack([
            torch.from_numpy(cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY))
            for img in images
        ]).to(device)
        # 批量预计算熵图
        full_entropy_maps = self._precompute_entropy(img_gray)  # [B,H,W]，虽然是计算整个熵图，但是是局部熵即基于像素相邻的，可以用于特征加权，强调边缘/纹理复杂的区域

        class_features = [[] for _ in range(self.num_classes)]  # 存储每个类别的特征，每个元素对应一个类别的特征列表
        entropy_weights = [[] for _ in range(self.num_classes)]  # 存储每个类别的熵权重，每个元素对应一个类别的权重列表

        for i in range(B):
            # 并行处理有效补丁
            valid_patches = self._get_valid_patches(img_gray[i], labels[i], full_entropy_maps[i], device)

            for y, x, size in valid_patches:
                # 提取特征
                feat_patch = feat[i, :, y:y + size, x:x + size]  # [C,size,size]
                feat_patch = feat_patch.permute(1, 2, 0).reshape(-1, C)  # [size², C]

                # 获取标签和权重
                patch_label = labels[i, y:y + size, x:x + size].flatten()  # [size²]
                yy, xx = torch.meshgrid(
                    torch.arange(y, y + size, device=device),
                    torch.arange(x, x + size, device=device)
                )
                point_weights = full_entropy_maps[i][yy, xx].flatten()  # [size²]

                # 分类收集
                for c in range(self.num_classes):
                    mask = (patch_label == c)
                    if mask.any():
                        class_feat = feat_patch[mask]  # 提取当前图像块中c类别的特征，class_feat包含了feat_patch中属于当前类别c的所有特征元素,[N_c, C]，N_c是补丁快中为c类的特征点个数
                        class_features[c].append(class_feat)  # 将当前图像块中c类别的特征块添加到对应类别的特征列表中,最终列表长度为筛选的补丁数量
                        entropy_weights[c].append(point_weights[mask])  # 每个特征点继承像素对应的熵权重，最终长度为该类所有特征块中特征点总数

        # 处理每个类别的特征和权重，是一个batchsize中所有的特征，也就是不同图片中的特征集合
        valid_features, valid_weights = [], []
        for c in range(self.num_classes):
            if len(class_features[c]) > 0:
                # 流式合并
                all_feats = torch.cat([f for f in class_features[c]], dim=0)  # 与entropy_weights每个列表长度相同，但是是(特征点总数,C）维度的
                all_weights = torch.cat([w for w in entropy_weights[c]], dim=0)  # (特征点总数,）维度

                # 仅做安全数量检查（防止内存爆炸）
                max_safe = min(len(all_feats), self.max_samples)  # 建议不超过3倍限制
                # 差异化处理不同类别
                if c == 1:  # 裂缝类别，保留全部特征
                    if len(all_feats) > max_safe:
                        _, idx = torch.topk(all_weights, max_safe)
                        all_feats = all_feats[idx]
                        all_weights = all_weights[idx]
                else:  # 背景类别，使用熵权重排序替代随机采样,优先保留高熵背景特征（可能与裂缝边缘相关)，，保留所有背景特征的前三分之的高熵区域
                    # 随机采样
                    if len(all_feats) > max_safe:

                        # 按熵权重排序选择高权重样本
                        _, indices = torch.topk(all_weights, max_safe)

                        all_feats = all_feats[indices]
                        all_weights = all_weights[indices]
                valid_features.append(all_feats)  # 随机采样特征添加到列表
                valid_weights.append(all_weights)

        if len(valid_features) < 2:  # 如果有效特征的数量小于2，则返回损失值为0
            return torch.tensor(0.0, device=device)

        # 修改后的对比损失计算（提升数值稳定性）
        total_loss = 0.0
        for c_idx, (anchors, weights) in enumerate(zip(valid_features, valid_weights)):
            N = anchors.size(0)
            if N < 2:
                continue

            # 计算动态温度参数（改进版本）
            if self.use_dynamic_temp:
                # 基于特征方差的温度调整（原实现，但使用更稳定的公式）
                feat_var = torch.var(anchors, dim=0).mean()  # 计算当前类别特征在每个维度上的方差，并取平均值
                dynamic_temp = self.temp_base + self.temp_scale * torch.sigmoid(feat_var - self.temp_center)
            else:
                dynamic_temp = self.temp

            # ================= 正样本计算 （类内）=================
            # 正样本计算（添加相似度偏移量）
            pos_sim = torch.mm(anchors, anchors.T)  # [N, N]，计算同一类别内样本之间的相似度矩阵
            pos_mask = ~torch.eye(N, dtype=torch.bool, device=device)  # 排除样本自身的相似度，即对角线元素
            # 正样本权重矩阵（使用外积）
            pos_weight_matrix = weights.unsqueeze(1) * weights.unsqueeze(0)  # [N, N]
            pos_weights = pos_weight_matrix[pos_mask]  # [N*(N-1), ]

            # 正损失（添加边界margin）
            pos_logits = (pos_sim[pos_mask] - 0.2) / dynamic_temp  # margin=0.2，将正样本对的相似度减去0.2后除以动态温度强制正样本对的相似度高于一般样本

            pos_loss_per_pair = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits),
                reduction='none') * pos_weights  # 计算每个正样本对的加权二元交叉熵损失，为困难样例挖掘做准备
            pos_loss = self._apply_ohem(pos_loss_per_pair, pos_weights, self.ohem_min_kept)  # 应用OHEM到正样本

            # ================= 负样本计算（类间） =================
            # 负样本计算（类间+类内）
            neg_samples = torch.cat([f for i, f in enumerate(valid_features) if i != c_idx])  # 拼接其他类别的特征作为负样本
            if len(neg_samples) == 0:
                continue
            neg_weights = torch.cat([f for i, f in enumerate(valid_weights) if i != c_idx])  # [M, ]

            neg_sim = torch.mm(anchors, neg_samples.T)  # [N, M]，计算当前类别样本与负样本之间的相似度矩阵
            neg_logits = (neg_sim + 0.1) / dynamic_temp  # margin=0.1，将负样本的相似度加上0.1后除以动态温度

            # 重新计算负样本权重矩阵，考虑所有负样本
            M = neg_samples.size(0)
            neg_weight_matrix = weights.unsqueeze(1) * neg_weights.unsqueeze(0)  # [N, M]
            neg_weights_flat = neg_weight_matrix.view(-1)  # [N*M, ]

            # 应用OHEM到负样本
            neg_loss_per_pair = F.binary_cross_entropy_with_logits(
                neg_logits.view(-1), torch.zeros_like(neg_logits.view(-1)),
                reduction='none') * neg_weights_flat
            neg_loss = self._apply_ohem(neg_loss_per_pair, neg_weights_flat, self.ohem_min_kept)

            total_loss += (pos_loss + 0.5 * neg_loss)  # 负样本权重系数，平衡正负样本数量差异，防止负样本主导优化方向

        if len(valid_features) > 0:
            total_loss /= len(valid_features)
        else:
            total_loss = torch.tensor(0.0, device=device)
        return total_loss