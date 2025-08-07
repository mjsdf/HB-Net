import argparse
import logging
import os
import pprint

import cv2

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, intersectionAndUnion1
from util.dist_helper import setup_distributed

from model.semseg.unet import UNet
# from util.patch import ContaminatedContrast
from util.patch import SemanticContrast

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


# 根据指定的评估模式（original、center_crop 或 sliding_window）对输入图像进行处理和模型预测。
# 计算预测结果与真实掩码的交集、并集等统计信息。
# 计算每个类别的交并比（IoU）和平均交并比（mIoU）。
# 对数据的标识符进行清理和规范化处理，并将预测标签保存到指定的路径。
# 最终返回平均交并比和每个类别的交并比。
def evaluate(epoch, model, loader, mode, cfg):
    """
       此函数用于评估模型在给定数据加载器和模式下的性能
       参数:
       - model: 要评估的模型
       - loader: 数据加载器，提供图像、掩码和标识符
       - mode: 评估模式，如 'original', 'center_crop', 'sliding_window'
       - cfg: 配置字典，可能包含有关评估的参数

       返回:
       - mIOU: 平均交并比
       - iou_class: 每个类别的交并比
       """
    model.eval()  # 将模型设置为评估模式
    assert mode in ['original', 'center_crop', 'sliding_window']  # 断言模式在指定的选项中
    intersection_meter = AverageMeter()  # 用于记录交集的平均值
    union_meter = AverageMeter()  # 用于记录并集的平均值
    # target_meter = AverageMeter()  # 新增目标统计
    intersection_meter1 = AverageMeter()  # 用于记录交集的平均值
    union_meter1 = AverageMeter()
    # output_meter = AverageMeter()  # 新增输出统计

    with torch.no_grad():  # 不计算梯度
        for img, mask, id in loader:  # 从加载器中迭代获取图像、掩码和标识符
            img = img.cuda()  # 将图像移动到 GPU
            # mask = mask.cuda()  # 将掩码也移动到 GPU

            if mode == 'sliding_window':  # 如果是滑动窗口模式
                grid = cfg['crop_size']  # 获取裁剪尺寸
                b, _, h, w = img.shape  # 获取图像形状,b为批次大小，h和w分别表示图像的高度和宽度，下划线表示一个不关心的维度
                final = torch.zeros(b, 2, h, w).cuda()  # 创建零张量用于累积预测
                row = 0
                while row < h:  # 按行进行处理
                    col = 0
                    while col < w:  # 按列进行处理
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])  # 模型预测
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)  # 累积预测
                        col += int(grid * 2 / 3)  # 列的移动步长
                    row += int(grid * 2 / 3)  # 行的移动步长

                # 经过 argmax(dim=1) 操作后，得到的是在通道维度（通常表示类别）上每个位置的最大类别索引。
                # 所以它得到的不是图片，而是一个与输入图像尺寸相同的索引数组，每个位置的索引表示该位置预测所属的类别。
                pred = final.argmax(dim=1)  # 获取最终预测结果

            else:
                if mode == 'center_crop':  # 如果是中心裁剪模式
                    h, w = img.shape[-2:]  # 获取图像高度和宽度
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2  # 计算裁剪起始位置
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]  # 进行裁剪
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]  # 对掩码也进行相应裁剪

                pred = model(img).argmax(dim=1)  # 模型预测

            # 计算交集、并集和目标
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            # 转换并归约统计量
            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)  # 分布式地归约交集
            dist.all_reduce(reduced_union)  # 分布式地归约并集
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())  # 更新交集的平均值
            union_meter.update(reduced_union.cpu().numpy())  # 更新并集的平均值

            del reduced_intersection, reduced_union, reduced_target  # 释放中间变量
            torch.cuda.empty_cache()

            # 计算原来的交集、并集和目标
            intersection1, union1, target1 = \
                intersectionAndUnion1(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            reduced_intersection1 = torch.from_numpy(intersection1).cuda()
            reduced_union1 = torch.from_numpy(union1).cuda()
            reduced_target1 = torch.from_numpy(target1).cuda()
            dist.all_reduce(reduced_intersection1)  # 分布式地归约交集
            dist.all_reduce(reduced_union1)  # 分布式地归约并集
            dist.all_reduce(reduced_target1)

            intersection_meter1.update(reduced_intersection1.cpu().numpy())  # 更新交集的平均值
            union_meter1.update(reduced_union1.cpu().numpy())  # 更新并集的平均值

            del reduced_intersection1, reduced_union1, reduced_target1  # 释放中间变量
            torch.cuda.empty_cache()

            # reduced_output = reduced_union1 + reduced_intersection1 - reduced_target1  # 计算预测输出量
            # dist.all_reduce(reduced_intersection1)  # 分布式地归约交集
            # intersection_meter1.update(reduced_intersection1.cpu().numpy())  # 更新交集的平均值
            #
            # target_meter.update(reduced_target1.cpu().numpy())  # 更新目标的平均值
            # output_meter.update(reduced_output.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0  # 计算每个类别的交并比
    mIOU = np.mean(iou_class)  # 计算平均交并比

    # 求原来的IoU
    iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10) * 100.0  # 计算每个类别的交并比
    mIOU1 = np.mean(iou_class1)  # 计算平均交并比

    # recall_class = intersection_meter1.sum / (target_meter.sum + 1e-10) * 100.0  # 计算每个类别的召回率,tp_meter.sum 是所有样本中每个类别的真正例数量之和，target_meter.sum 是所有样本中每个类别的真实标签数量之和
    # precision_class = intersection_meter1.sum / (output_meter.sum + 1e-10) * 100.0  # 计算每个类别的精确率
    # f1_class = 2 * precision_class * recall_class / (precision_class + recall_class + 1e-10)  # 计算每个类别的 F1 分数

    return mIOU, iou_class, mIOU1, iou_class1


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet(in_chns=3, class_num=cfg['nclass'])
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    # 根据配置创建有标签或无标签的损失函数
    if cfg['criterion']['name'] == 'CELoss':
        # 手动指定背景类和前景裂缝的权重
        weight_tensor = torch.tensor([1.0, 10.0]).cuda(local_rank)  # 假设背景权重为 1.0，前景裂缝权重为 10.0
        criterion_l = nn.CrossEntropyLoss(weight=weight_tensor, **cfg['criterion']['kwargs']).cuda(local_rank)
        # criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)

    elif cfg['criterion']['name'] == 'OHEM':  # 监督损失为OHEM损失
        weight_tensor = torch.tensor([1.0, 10.0]).cuda(local_rank)
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)

    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latestsup101.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latestsup101.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_contrast = AverageMeter()  # 对比扰动损失

        trainsampler.set_epoch(epoch)

        # 根据不同阶段设置学习率
        if epoch < 15:  # 多项式衰减策略，训练初期缓慢下降，使模型学习到数据的基本特征
            lr = cfg['lr'] * (1 - epoch / 20) ** 0.9
        elif 15 <= epoch < 50:  # 直接降至基础的10%，避免因学习率过大而导致模型在接近最优解时跳过最优解
            lr = cfg['lr'] * 0.1
        else:  # 线性衰减策略
            lr = cfg['lr'] * 0.01 * (1 - epoch / 100)
        optimizer.param_groups[0]["lr"] = lr

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            if epoch >= 15:
                # 调用 enable_projection 方法初始化注意力模块
                if not model.module.return_attn:
                    model.module.enable_projection()
                pred, feat = model(img, True, return_attn=True)
                final_feat = feat
            else:
                pred = model(img, True, return_attn=False)
            preds, preds_fp = pred.chunk(2)

            loss_x = criterion_l(preds, mask)
            if epoch >= 15:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    # contrast_loss = model.module.get_contrast_loss(final_feat,  torch.cat((img_x, img_u_w)))
                    contrast_loss = model.module.get_contrast_loss(final_feat, img, mask)
                else:
                    # contrast_loss = model.get_contrast_loss(final_feat,  torch.cat((img_x, img_u_w)))
                    contrast_loss = model.get_contrast_loss(final_feat, img, mask)
            else:
                contrast_loss = 0

            # loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + 0.3 * contrast_loss) / 2.0  # 计算总的损失
            # 根据不同阶段计算总损失
            if epoch < 15:
                loss = loss_x

            else:
                # loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + contrast_weight * contrast_loss) / 2.0
                loss = (loss_x + 0.5 * contrast_loss) / 1.5

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            if isinstance(contrast_loss, torch.Tensor):
                total_loss_contrast.update(contrast_loss.item())
            else:
                total_loss_contrast.update(contrast_loss)

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr  # 更新优化器的学习率

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
                if isinstance(contrast_loss, torch.Tensor):
                    writer.add_scalar('train/loss_contrast', contrast_loss.item(), iters)
                else:
                    writer.add_scalar('train/loss_contrast', contrast_loss, iters)

            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f},  Loss loss_x: {:.3f},  Loss contrast: {:.3f}'.format(i,
                                                                                                                  total_loss.avg,
                                                                                                                  total_loss_x.avg,
                                                                                                                  total_loss_contrast.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class, mIoU1, iou_class1 = evaluate(epoch, model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou, iou1) in zip(range(len(iou_class)), iou_class, iou_class1):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f},IoU1:{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou1))
            logger.info(
                '***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, MeanIoU1: {:.2f}\n'.format(eval_mode, mIoU, mIoU1))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU1', mIoU1, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU1' % (CLASSES[cfg['dataset']][i]), iou1, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latestsup101.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'bestsup101.pth'))


if __name__ == '__main__':
    main()
