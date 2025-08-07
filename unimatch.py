import argparse
import logging
import os
import pprint
import numpy as np

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# 设置环境变量以解决 OpenMP 问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# 导入自定义模块
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

from model.semseg.unet import UNet
# from util.patch import ContaminatedContrast
from util.patch import SemanticContrast

# 这个脚本实现了半监督语义分割的训练流程，包括数据集的加载、模型的初始化、优化器的设置、训练循环、损失函数的计算和评估过程。
# 使用了分布式数据并行（DistributedDataParallel）来加速训练。
# 支持两种损失函数：交叉熵损失（CELoss）和在线难例挖掘损失（OHEM）。
# 训练过程中使用了混合图像技术（CutMix）和教师模型的软标签来增强无标签数据的学习。
# 评估函数 evaluate 计算了模型在验证集上的平均交并比（mIoU）。
# 训练和评估的结果会被记录到日志文件和 TensorBoard 中。
# 模型的检查点会在每个 epoch 结束时保存，并在达到新的最好性能时更新最佳模型的保存文件。


# 设置命令行参数解析
parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()  # 解析命令行参数

    # cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)  # 加载配置文件，即Pascal.yaml文件。原始的代码，在我将liefeng.yaml中的损失函数改成OHEM后，又改回来后，这里就需要加encoding='utf-8'才不报错
    cfg = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.Loader)  # 加载配置文件，即Pascal.yaml文件
    # with open(args.config, 'r', encoding='utf-8') as f:   #我将liefeng.yaml中的损失函数改成OHEM后需要改这里
    #     cfg = yaml.load(f, Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)  # 初始化日志记录器
    logger.propagate = 0  # 禁止日志向上传播到父级记录器，避免重复记录

    rank, world_size = setup_distributed(port=args.port)  # 设置分布式环境

    if rank == 0:  # 如果当前进程的排名为 0（通常为主进程）
        all_args = {**cfg, **vars(args), 'ngpus': world_size}  # 日志记录所有参数
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)  # 初始化 TensorBoard 记录器，用于记录训练过程中的数据，保存路径为命令行指定的 args.save_path

        os.makedirs(args.save_path, exist_ok=True)  # 确保保存路径存在


    cudnn.enabled = True  # 启用 CUDNN 加速
    cudnn.benchmark = True  # 启用 CUDNN 的自动调优功能，以提高性能

    # # model = DeepLabV3Plus(cfg)  # 创建 DeepLabV3Plus 模型实例
    # model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet}
    # assert cfg['model'] in model_zoo.keys()
    # model = model_zoo[cfg['model']](cfg)
    #
    # # 创建随机梯度下降优化器SGD，两个参数组，第一个参数组表示对backbone的参数使用较小的学习率lr微调，避免过度修改预训练的特征提取部分
    # # 第二个参数组是模型中除骨干部分的其他参数，需要更大的学习率lr_multi调整，这样可以为不同部分的模型参数设置不同的学习率
    # optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
    #                  {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
    #                   'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    # UNet的定义
    model = UNet(in_chns=3, class_num=cfg['nclass'])
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    # 仅在 rank 0 上打印参数数量
    # 行计算框架中（例如 MPI），会有多个进程同时运行相同或不同的任务。每个进程都有一个唯一的rank值，用于区分不同的进程
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # 将 SyncBN 与分布式模型结合，可以在多个 GPU 之间有效地同步归一化参数的同时，还能实现模型其他参数的高效并行训练
    # 在分布式训练中，当模型在多个GPU或计算节点上并行训练时，不同设备上的批次数据可能存在差异，这会导致标准批归一化的统计信息（均值和方差）在不同设备上有所不同。
    # 而同步批归一化通过在多个GPU设备之间同步归一化参数，可以解决这个问题
    local_rank = int(os.environ["LOCAL_RANK"])  # 从环境变量中获取本地排名，并将其转换为整数
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 将模型中的批归一化层转换为同步批归一化层，用于分布式训练
    model.cuda()  # 将模型移动到 GPU 上进行计算

    # 包装模型为分布式模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    # 根据配置创建有标签或无标签的损失函数
    if cfg['criterion']['name'] == 'CELoss':
        # 手动指定背景类和前景裂缝的权重
        weight_tensor = torch.tensor([1.0, 10.0]).cuda(local_rank)  # 假设背景权重为 1.0，前景裂缝权重为 10.0
        criterion_l = nn.CrossEntropyLoss(weight=weight_tensor, **cfg['criterion']['kwargs']).cuda(local_rank)

    elif cfg['criterion']['name'] == 'OHEM':  # 监督损失为OHEM损失
        weight_tensor = torch.tensor([1.0, 10.0]).cuda(local_rank)
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)

    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    # 创建无标签交叉熵损失函数，设置 reduction='none' 表示不进行任何归约操作，并将其移动到指定的 GPU（由 local_rank 决定）
    criterion_u = nn.CrossEntropyLoss(weight=weight_tensor, reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    # 创建分布式采样器和数据加载器
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    # 初始化训练迭代次数和最佳 mIoU
    total_iters = len(trainloader_u) * cfg['epochs']  # 计算总的迭代次数
    previous_best = 0.0
    epoch = -1

    # 指定路径存在保存的检查点文件latest.pth时，从中恢复模型的参数、优化器的状态、训练的轮数以及之前的最佳性能指标等信息，以便继续训练或基于之前的训练结果进行进一步的操作
    if os.path.exists(os.path.join(args.save_path, 'latest311-3.pth')):  # 检查指定路径下是否存在名为 'latest.pth' 的文件
        # 添加代码
        model.module.enable_projection()  # 强制启用投影头

        checkpoint = torch.load(os.path.join(args.save_path, 'latest311-3.pth'))  # 如果文件存在，加载这个文件的内容到 checkpoint 变量中
        model.load_state_dict(checkpoint['model'], strict=False)  # 从 checkpoint 中获取 'model' 对应的模型参数，并加载到当前模型中
        optimizer.load_state_dict(checkpoint['optimizer'])  # 从 checkpoint 中获取 'optimizer' 对应的优化器参数，并加载到当前优化器中
        epoch = checkpoint['epoch']  # 从 checkpoint 中获取 'epoch' 的值，可能用于记录训练的轮数
        previous_best = checkpoint['previous_best']  # 从 checkpoint 中获取 'previous_best' 的值，可能用于记录之前的最佳性能指标

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    # 进行一轮轮epoch的训练和评估
    # 每轮训练中，从有标签和无标签的数据加载器中获取数据，进行前向传播、计算损失、反向传播和优化器更新。
    # 同时记录各种损失值和掩码比例，并根据迭代次数调整学习率。
    # 在一定间隔打印训练信息。训练完一轮后，根据设置的评估模式对模型进行评估，记录评估结果(包括每个类别的IoU和平均IoU)到日志和TensorBoard中。
    # 还会检查当前模型是否是最佳模型，并保存相应的检查点。
    # 整个过程在分布式环境下进行，通过判断进程rank来控制主进程的操作。
    for epoch in range(epoch + 1, cfg['epochs']):  # 从上次中断的轮数开始，迭代到配置的总轮数
        if rank == 0:  # 如果是主进程
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))  # 打印日志信息

        total_loss = AverageMeter()  # 记录总损失
        total_loss_x = AverageMeter()  # 监督损失
        total_loss_s = AverageMeter()  # 强扰动损失
        total_loss_w_fp = AverageMeter()  # 特征扰动损失
        total_loss_contrast = AverageMeter()  # 对比扰动损失
        total_mask_ratio = AverageMeter()  # 掩码比例

        trainloader_l.sampler.set_epoch(epoch)  # 为有标签数据的采样器设置轮数
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)  # 将有标签和无标签的加载器组合

        # 根据不同阶段设置学习率
        if epoch < 15:  # 多项式衰减策略，训练初期缓慢下降，使模型学习到数据的基本特征
            lr = cfg['lr'] * (1 - epoch / 20) ** 0.9
        elif 15 <= epoch < 50:  # 直接降至基础的10%，避免因学习率过大而导致模型在接近最优解时跳过最优解
            lr = cfg['lr'] * 0.1
        else:  # 线性衰减策略
            lr = cfg['lr'] * 0.01 * (1 - epoch / 100)
        optimizer.param_groups[0]["lr"] = lr

        # loader 是一个可以迭代的对象，每次迭代时会从loader中取出一个元素，该元素包含三个元组
        # 实现在一次迭代epoch中同时获取和处理有标签数据和多种无标签数据的不同形式
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()  # 将有标签数据移动到GPU

            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()  # 两个强增强用同一个mask
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()  # 将相关参数移动到GPU

            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():  # 不需要计算梯度的上下文
                model.eval()  # 模型设置为评估模式

                pred_u_w_mix = model(img_u_w_mix, False, return_attn=False).detach()  # 进行预测并分离梯度
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]  # 计算置信度
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)  # 获取预测的掩码

            # 根据cutmix_box1确定的条件，将img_u_s1中的某些位置的元素用img_u_s1_mix中对应位置的元素进行替换
            # 即进行CutMix增强
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()  # 模型设置为训练模式

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]  # 获取有标签和无标签数据的数量

            # pred, final_feat = model(torch.cat((img_x, img_u_w)), True, return_attn=True)  # 进行模型预测
            if epoch >= 15:
                # 调用 enable_projection 方法初始化注意力模块
                if not model.module.return_attn:
                    model.module.enable_projection()
                pred, feat = model(torch.cat((img_x, img_u_w)), True, return_attn=True)
                final_feat = feat[:num_lb]
            else:
                pred = model(torch.cat((img_x, img_u_w)), True, return_attn=False)
            preds, preds_fp = pred.chunk(2)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])  # 拆分预测结果
            pred_u_w_fp = preds_fp[num_lb:]  # 获取特定部分的预测结果

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), False, return_attn=False).chunk(
                2)  # 进行模型预测并拆分

            pred_u_w = pred_u_w.detach()  # 分离无标签图像预测结果的梯度，防止在后续计算中对模型参数产生影响
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]  # 计算无标签图像预测结果的置信度
            mask_u_w = pred_u_w.argmax(dim=1)  # 获取预测的掩码

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()  # 克隆相关数据，保留原始数据的副本
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            # 数据的混合操作，类似于CutMix，当cutmix_box1中的元素为1时，将mask_u_w_mix中对应位置的元素赋值给mask_u_w_cutmixed1
            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)  # 计算有标签数据的损失

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()  # 计算平均损失

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            if epoch >= 15:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    contrast_loss = model.module.get_contrast_loss(final_feat, img_x, mask_x)
                else:
                    contrast_loss = model.get_contrast_loss(final_feat, img_x, mask_x)
            else:
                contrast_loss = 0

            # 计算总的损失
            # 根据不同阶段计算总损失
            if epoch < 15:
                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            else:
                # loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + contrast_weight * contrast_loss) / 2.0
                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + 0.3 * contrast_loss) / 2.5

            torch.distributed.barrier()  # 分布式同步操作

            optimizer.zero_grad()  # 优化器梯度清零
            loss.backward()  # 损失反向传播
            optimizer.step()  # 优化器执行一步更新

            total_loss.update(loss.item())  # 更新总的损失记录
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            # total_loss_contrast.update(contrast_loss.item())
            # 修改此处，添加类型检查
            if isinstance(contrast_loss, torch.Tensor):
                total_loss_contrast.update(contrast_loss.item())
            else:
                total_loss_contrast.update(contrast_loss)

            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                         (ignore_mask != 255).sum()  # 计算掩码比例
            total_mask_ratio.update(mask_ratio.item())  # 更新掩码比例记录

            iters = epoch * len(trainloader_u) + i  # 计算当前的迭代次数
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9  # 计算学习率
            optimizer.param_groups[0]["lr"] = lr  # 更新优化器的学习率
            # 因为Unet里没有backbone，所以不用学习率
            # optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)  # 将损失记录到 TensorBoard

                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                # writer.add_scalar('train/loss_contrast', contrast_loss.item(), iters)
                if isinstance(contrast_loss, torch.Tensor):
                    writer.add_scalar('train/loss_contrast', contrast_loss.item(), iters)
                else:
                    writer.add_scalar('train/loss_contrast', contrast_loss, iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            # len(trainloader_u)表示数据加载器trainloader_u中的批次数，即trainset_u//batch_size
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):  # 每隔一定间隔且是主进程
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f},  Loss contrast: {:.3f}, Mask ratio: '
                    '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                    total_loss_w_fp.avg, total_loss_contrast.avg, total_mask_ratio.avg))  # 打印日志信息

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'  # 根据数据集设置评估模式
        mIoU, iou_class, mIoU1, iou_class1 = evaluate(epoch, model, valloader, eval_mode, cfg)  # 进行模型评估

        if rank == 0:
            # 日志记录每个类别的 IoU 和平均 mIoU
            for (cls_idx, iou, iou1) in zip(range(len(iou_class)), iou_class, iou_class1):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f},IoU1:{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou1))
            logger.info(
                '***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, MeanIoU1: {:.2f}\n'.format(eval_mode, mIoU, mIoU1))

            # 将评估结果记录到 TensorBoard
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU1', mIoU1, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU1' % (CLASSES[cfg['dataset']][i]), iou1, epoch)

        # 检查是否是最佳模型，并保存检查点
        is_best = mIoU1 > previous_best
        previous_best = max(mIoU1, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest311-3.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best311-3.pth'))


if __name__ == '__main__':
    main()
