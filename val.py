import torch
import yaml
from unimatch import UNet, SemiDataset, evaluate, CLASSES, setup_distributed
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # 这里直接指定配置文件的路径，你可以根据实际情况修改这个路径
    config_file_path = "D:/xiangmu/PaperCode/UniMatch-main/configs/liefeng30.yaml"
    # 指定utf-8编码打开文件，再加载配置内容
    with open(config_file_path, "r", encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # 设置分布式环境（和原训练代码保持一致的设置逻辑），这里简化处理，暂不考虑分布式相关复杂配置，可根据实际需求完善
    rank = 0
    world_size = 1

    # 加载最佳模型的参数（从保存的best.pth文件中加载），这里假设best.pth在指定的保存路径下，你可根据实际调整
    save_path = "D:/xiangmu/PaperCode/UniMatch-main/exp/liefeng30/unimatch/unet/160labeled"
    # save_path = "D:/xiangmu/PaperCode/UniMatch-main/exp/liefeng30/supervised"
    checkpoint = torch.load(os.path.join(save_path, 'best311.pth'))
    # 先获取保存的模型权重状态字典
    state_dict = checkpoint['model']
    # 处理键名，去除 "module." 前缀（因为保存时是分布式训练包装后的模型，键名有前缀）
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 创建模型实例
    model = UNet(in_chns=3, class_num=cfg['nclass'])
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()

    # 将模型设置为评估模式
    model.eval()
    # 创建验证数据集对象（和原训练代码中验证数据集的创建逻辑一致）
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    # 根据数据集设置评估模式（和原训练代码中的逻辑一致）
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'

    def get_class_color(class_idx, nclass):
        if class_idx == 0:
            return np.array([0, 0, 0])
        else:
            return np.array([255, 255, 255])

    # 定义保存预测结果的目录，这里使用和代码文件同级的目录下的 'best_prediction_results' 文件夹来保存，你可以根据需求修改
    save_dir = os.path.join(r'D:\xiangmu\PaperCode\UniMatch-main\best_val_predict', 'best311_val_predict')
    os.makedirs(save_dir, exist_ok=True)

    # 不计算梯度，进行预测并保存结果
    # 常用于模型推理阶段，节省内存并加快计算速度，因为推理时不需要计算梯度来更新模型参数
    with torch.no_grad():
        # 遍历验证集数据加载器valloader，每次迭代会返回图像数据img、对应的掩码数据mask以及图像的标识id
        for img, mask, id in valloader:
            # 将图像数据移动到GPU上进行加速计算（前提是已经配置好GPU环境且支持CUDA）
            img = img.cuda()

            # 判断评估模式是否为滑动窗口模式
            if eval_mode == 'sliding_window':
                # 获取裁剪尺寸，从配置文件（假设cfg是配置相关的字典）中读取
                grid = cfg['crop_size']
                # 获取当前批次图像数据的形状信息，b表示批次大小，_表示通道数（这里暂时不关心），h表示图像高度，w表示图像宽度
                b, _, h, w = img.shape
                # 创建一个全零的张量，用于存储最终的预测结果，形状与输入图像对应，通道数为2（可能对应不同的类别预测结果），并将其移动到GPU上
                final = torch.zeros(b, 2, h, w).cuda()
                # 初始化行索引为0，用于滑动窗口在图像高度方向上的滑动
                row = 0
                # 开始按行滑动窗口，只要当前行索引小于图像高度，就继续循环
                while row < h:
                    # 初始化列索引为0，用于滑动窗口在图像宽度方向上的滑动
                    col = 0
                    # 开始按列滑动窗口，只要当前列索引小于图像宽度，就继续循环
                    while col < w:
                        # 使用模型对当前滑动窗口截取的图像区域进行预测，截取图像时确保不超出图像边界
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        # 将当前窗口的预测结果累加到最终结果张量final中，这里先进行softmax操作，将模型输出转换为概率分布形式（假设是分类任务）
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        # 按照一定步长（这里是grid * 2 / 3）移动列索引，实现滑动窗口在宽度方向上的滑动
                        col += int(grid * 2 / 3)
                    # 按照一定步长（这里是grid * 2 / 3）移动行索引，实现滑动窗口在高度方向上的滑动
                    row += int(grid * 2 / 3)

                # 获取最终预测结果中概率最大的类别索引，将其转换为numpy数组，并将数据从GPU移回CPU内存
                pred = final.argmax(dim=1).cpu().numpy()

            else:
                # 判断评估模式是否为中心裁剪模式
                if eval_mode == 'center_crop':
                    # 获取图像的高度和宽度，这里取图像形状中的后两个维度
                    h, w = img.shape[-2:]
                    # 计算在高度方向上的裁剪起始位置，确保裁剪出的区域是以图像中心为基准的指定大小（从配置文件中读取裁剪尺寸cfg['crop_size']）
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    # 对图像进行中心裁剪，获取裁剪后的图像数据
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    # 对掩码数据也进行同样的中心裁剪操作，确保与图像裁剪后的数据对应（如果需要的话）
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                # 使用模型对处理后的图像（经过滑动窗口或者中心裁剪后的图像）进行预测，获取预测结果中概率最大的类别索引，
                # 然后将其转换为numpy数组，并将数据从GPU移回CPU内存
                pred = model(img).argmax(dim=1).cpu().numpy()

            # 遍历当前批次中的每一个预测结果（对应每一张图像的预测情况）
            for i in range(pred.shape[0]):
                # 将图像的标识字符串按空格分割，得到多个部分（具体格式取决于数据集中的标识设置）
                id_parts = id[i].split(' ')
                # 从标识的第一个部分中提取文件名，通过先按'/'分割取第二个元素（去掉路径部分），再按'.png'分割取第一个元素（去掉后缀）
                file_name = id_parts[0].split('/')[1].split('.png')[0]
                # 构建完整的图像文件名，加上后缀.png
                image_name = f'{file_name}.png'
                # 获取当前图像的预测结果（已经是类别索引形式的numpy数组）
                prediction = pred[i]
                # 创建一个全零的numpy数组，用于存储将类别索引转换为RGB颜色表示后的预测结果图像，
                # 形状根据预测结果的高度、宽度以及RGB三个通道来设置，数据类型为无符号8位整数（适合存储图像像素值）
                rgb_prediction = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
                # 遍历每个类别索引，根据类别索引将对应的像素位置设置为相应的颜色值（通过调用get_class_color函数获取颜色）
                for class_idx in range(cfg['nclass']):
                    rgb_prediction[prediction == class_idx] = get_class_color(class_idx, cfg['nclass'])
                # 使用OpenCV库将RGB颜色表示的预测结果图像保存到指定的目录下（save_dir是保存目录的路径），文件名就是前面构建的image_name
                cv2.imwrite(os.path.join(save_dir, image_name), rgb_prediction)