import argparse
import os
import pprint

import numpy as np
import torch
import yaml
import logging

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset.semi import SemiDataset
from model.semseg.unet import UNet
from util.classes import CLASSES
from util.utils import init_log
from util.dist_helper import setup_distributed

def test(model, test_loader, device, logger, save_path):
    model.eval()
    total_iou = 0.0
    iou_per_class = {cls: 0.0 for cls in CLASSES[cfg['dataset']]}
    num_images = len(test_loader.dataset)

    os.makedirs(save_path, exist_ok=True)  # 创建保存预测结果的目录，如果已存在则不报错

    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            pred = output.argmax(dim=1)

            for cls in CLASSES[cfg['dataset']]:
                cls_mask = (mask == CLASSES[cfg['dataset']].index(cls))
                cls_pred = (pred == CLASSES[cfg['dataset']].index(cls))
                intersection = (cls_mask & cls_pred).sum().item()
                union = (cls_mask | cls_pred).sum().item()
                if union > 0:
                    iou_per_class[cls] += intersection / union
                else:
                    iou_per_class[cls] += 1.0

            intersection = (mask == pred).sum().item()
            union = (mask!= 0).sum().item() + (pred!= 0).sum().item() - intersection
            if union > 0:
                total_iou += intersection / union
            else:
                total_iou += 1.0

            # 将预测结果转换为numpy数组，并进行可视化处理（这里简单地将预测结果转换为灰度图像，类别0为黑色，类别1为白色）
            pred_np = pred.cpu().numpy().astype(np.uint8) * 255
            img_name = os.path.splitext(os.path.basename(test_loader.dataset.ids[i]))[0]  # 获取原始图像的文件名（不含扩展名）
            save_img_path = os.path.join(save_path, img_name + '.png')  # 构建保存预测结果图像的路径
            plt.imsave(save_img_path, pred_np, cmap='gray')  # 保存预测结果图像

    mean_iou = total_iou / num_images
    mean_iou_per_class = {cls: iou / num_images for cls, iou in iou_per_class.items()}

    logger.info('***** Test Results *****')
    logger.info('Mean IoU: {:.2f}'.format(mean_iou * 100))
    for cls, iou in mean_iou_per_class.items():
        logger.info('Class [{:}] IoU: {:.2f}'.format(cls, iou * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained model on the test set')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test-id-path', type=str, default='D:/xiangmu/PaperCode/UniMatch-main/splits/liefeng3/train.txt')  # 设置测试集路径，可根据实际修改
    parser.add_argument('--save-path', type=str, default='D:/xiangmu/PaperCode/UniMatch-main/exp/liefeng2-1-1/unimatch/unet/50labeled')  # 设置模型保存路径，可根据实际修改
    parser.add_argument('--result-save-path', type=str, default='D:/xiangmu/PaperCode/UniMatch-main/test')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=15020, type=int)

    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.Loader)# 加载配置文件，即Pascal.yaml文件

    logger = init_log('global', logging.INFO) # 初始化一个全局日志记录器，日志级别为INFO，用于记录训练和测试过程中的信息
    logger.propagate = 0 # 禁止日志向上传播到父级记录器，避免重复记录
    # 直接设置设备为cuda:0（如果有GPU）或cpu（如果没有GPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # rank, world_size = setup_distributed(port=args.port) # 设置分布式训练环境，根据指定的端口（由命令行参数指定）获取当前进程的排名（rank）和总的进程数量（world_size）

    # 如果当前进程的排名为0（通常是主进程），则打印配置信息，格式化输出配置字典的内容
    # if rank == 0:
    #     logger.info('{}\n'.format(pprint.pformat(cfg)))

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        local_rank = 0  # 在非分布式环境下，默认使用设备0（可以根据实际情况调整，如果没有GPU则可以设置为 'cpu'）
    device = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')  # 从环境变量中获取本地排名，并将其转换为整数
    device = torch.device('cuda:{}'.format(local_rank))

    model = UNet(in_chns=3, class_num=2)# 创建一个UNet模型实例，输入通道数为3（通常对应RGB图像），类别数量由配置文件中的'nclass'指定
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    checkpoint = torch.load(os.path.join(args.save_path, 'best1first.pth')) # 从指定的保存路径（由命令行参数指定）加载训练好的模型的检查点文件（best1first.pth）
    model.load_state_dict(checkpoint['model']) # 将检查点中的模型参数加载到当前模型中，使模型恢复到训练好的状态

    testset = SemiDataset(cfg['dataset'], cfg['data_root'], 'test', args.test_id_path)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    test_loader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1,
                             drop_last=False, sampler=test_sampler)

    test(model, test_loader, device, logger, args.result_save_path)