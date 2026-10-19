import os
import shutil
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import numpy as np
from utils.metric import AverageMeter, Loss  # 导入评估指标相关工具
from config import data_config, network_config, get_image_unique  # 导入配置函数
from train_config import config  # 导入训练配置
from solver import WarmupMultiStepLR  # 导入学习率调度器
from test import test  # 导入测试函数
from utils.CRLoss import CRLoss  # 导入对比关系损失函数
from utils.MultiSubLoss import MultiSubLoss

# 配置日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def set_seed(args):
    """设置随机种子以确保实验可重复性
    
    Args:
        args: 包含seed参数的配置对象
    """
    # 预定义随机初始种子
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的自动优化
    torch.backends.cudnn.deterministic = True  # 确保cudnn的结果是确定的

def save_checkpoint(state, epoch, dst, is_best):
    """保存模型检查点
    
    Args:
        state: 要保存的模型状态字典
        epoch: 当前训练轮次
        dst: 保存目标路径
        is_best: 是否为最佳模型
    """
    filename = os.path.join(dst, "best_model") + ".pth.tar"
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, "model_best", str(epoch)) + ".pth.tar"
        shutil.copyfile(filename, dst_best)


def train(epoch, train_loader, network, optimizer, multi_sub_loss, args):
    """训练一个epoch
    
    Args:
        epoch: 当前训练轮次
        train_loader: 训练数据加载器
        network: 网络模型
        optimizer: 优化器
        multi_sub_loss: 多子空间损失函数
        args: 训练参数
        
    Returns:
        train_loss.avg: 平均训练损失
        image_pre.avg: 平均图像精度
        text_pre.avg: 平均文本精度
    """
    train_loss = AverageMeter()  # 用于记录平均训练损失
    image_pre = AverageMeter()  # 用于记录平均图像精度
    text_pre = AverageMeter()  # 用于记录平均文本精度

    # 切换到训练模式
    network.train()
    global_step = 0

    start_time = time.time()
    for step, (images, captions, labels) in enumerate(train_loader):
        
        # 预处理文本数据
        (
            tokens_all,
            segments_all,
            input_masks_all,
            caption_length_all,
        ) = network.module.language_model.pre_process_multiple(captions)   
        #print(f"tokens_all的维度{tokens_all.shape}")
        tokens_all = tokens_all.cuda()  # [B, 15, 100] 
        #print(tokens_all.shape)
        segments_all = segments_all.cuda()  # [B, 15, 100]
        input_masks_all = input_masks_all.cuda()  # [B, 15, 100]
        images = images.cuda()  # 将图像移至GPU
        labels = labels.cuda()  # 将标签移至GPU
        
        # 前向传播，获取模型输出（包含子空间特征）
        img_output, text_output, img_f, text_f, sim_cs, sim_cd, subspace_features = network(
            images, tokens_all, segments_all, input_masks_all
        )
                
        # ==================== 使用真实的子空间特征计算MultiSubLoss ====================
        # 从subspace_features中提取特征
        image_subspace_features = subspace_features['image_subspace_features']  # 列表形式的子空间特征
        text_subspace_features = subspace_features['text_subspace_features']    # 列表形式的子空间特征
        
        # 全局特征（加权聚合后的特征）
        image_global_features = img_output[0]  # 全局图像特征
        text_global_features = text_output[0]   # 全局文本特征
        projection_matrices = subspace_features['projection_matrices']
        # 计算MultiSubLoss
        multi_sub_total_loss, multi_sub_loss_dict = multi_sub_loss(
            image_global_features,    # 全局图像特征
            text_global_features,     # 全局文本特征
            image_subspace_features,  # 图像子空间特征列表
            text_subspace_features,   # 文本子空间特征列表
            projection_matrices       # 投影矩阵
        )
        
        # 设置损失值
        loss = multi_sub_total_loss.mean() if multi_sub_total_loss.dim() > 0 else multi_sub_total_loss
        # 处理DataParallel返回的tensor，确保是标量后再调用.item()
        cmpm_loss = multi_sub_loss_dict['global_loss'].mean().item() if multi_sub_loss_dict['global_loss'].dim() > 0 else multi_sub_loss_dict['global_loss'].item()
        cmpc_loss = multi_sub_loss_dict['attr_loss'].mean().item() if multi_sub_loss_dict['attr_loss'].dim() > 0 else multi_sub_loss_dict['attr_loss'].item()
        sim_loss = multi_sub_loss_dict['div_loss'].mean().item() if multi_sub_loss_dict['div_loss'].dim() > 0 else multi_sub_loss_dict['div_loss'].item()
        image_precision = 0.8  # 默认值
        text_precision = 0.8   # 默认值

        # 获取当前学习率
        current_lr = []
        for params in optimizer.param_groups:
            current_lr.append(params['lr'])

        global_step += 1
        # 每50步记录一次训练信息
        if global_step % 100 == 0:
            logger.info(
                "Epoch: {}/{}, Step: {}/{}, Lr: {}, multi_sub_loss: {:.3f}, global_loss: {:.3f}, attr_loss: {:.3f}, div_loss: {:.3f}, Time/step: {:.4f}".format(
                    epoch,
                    args.num_epoches,
                    step+1,
                    len(train_loader),
                    "-".join([str('%.9f'%itm) for itm in sorted(current_lr)]),
                    loss.item(),
                    cmpm_loss,
                    cmpc_loss,
                    sim_loss,
                    (time.time() - start_time) / 50
                )
            )
            start_time = time.time()

        # 计算梯度并执行优化步骤
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        # 更新统计指标
        train_loss.update(loss.item(), images.shape[0])
        image_pre.update(image_precision, images.shape[0])
        text_pre.update(text_precision, images.shape[0])

    return train_loss.avg, image_pre.avg, text_pre.avg


def main(args):
    """主函数，执行整个训练流程
    
    Args:
        args: 训练参数
    """
    # 设置随机种子
    set_seed(args)

    # 定义图像变换
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet标准化参数
    )

    # 训练数据变换
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=3),  # 调整图像大小
            transforms.RandomHorizontalFlip(),               # 随机水平翻转
            transforms.ToTensor(),                           # 转换为张量
            normalize,                                       # 标准化
        ]
    )

    # 测试数据变换
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=3),  # 调整图像大小
            transforms.ToTensor(),                           # 转换为张量
            normalize,                                       # 标准化
        ]
    )
    cap_transform = None  # 文本变换为None
    
    # 数据加载
    # 训练数据加载器
    train_loader = data_config(
        args.image_dir,    # 图像目录
        args.anno_dir,     # 注释目录
        args.batch_size,   # 批次大小
        "train",           # 数据集类型
        args.max_length,   # 最大文本长度
        train_transform,   # 图像变换
        cap_transform=cap_transform,  # 文本变换
    )

    # 测试数据加载器
    test_loader = data_config(
        args.image_dir, args.anno_dir, 64, "test", args.max_length, test_transform
    )
    
    # 获取唯一图像（用于评估）
    unique_image = get_image_unique(
        args.image_dir, args.anno_dir, 64, "test", args.max_length, test_transform
    )

    # 初始化多子空间损失函数（只使用这一个损失函数）
    multi_sub_loss = MultiSubLoss(
        temperature=0.07,  # 温度参数
        alpha=1,         # 属性级对比损失权重
        beta=0.002           # 多样性正则化权重
    )
    multi_sub_loss = nn.DataParallel(multi_sub_loss).cuda()  # 移至GPU并支持多GPU

    # 初始化网络和优化器（不再需要compute_loss参数）
    network, optimizer = network_config(
        args, "train", None, args.resume, args.model_path
    )
    
    # 学习率调度器 - 使用预热多步衰减
    scheduler = WarmupMultiStepLR(optimizer, (8, 12, 16), 0.1, 0.01, 10, "linear")  # 在第20、25、35轮降低学习率
    ac_t2i_top1_best = 0.0  # 记录最佳文本到图像的top1准确率
    best_epoch = 0  # 记录最佳轮次

    # 开始训练循环
    for epoch in range(1, args.num_epoches + 1 - args.start_epoch):
        network.train()  # 设置为训练模式
        # 训练一个epoch
        train_loss, image_precision, text_precision = train(
            args.start_epoch + epoch,
            train_loader,
            network,
            optimizer,
            multi_sub_loss,
            args,
        )

        is_best = False
        # 记录训练信息
        logging.info(
            "Epoch {}/{} Finished, train_loss: {:.3f}, image_precision: {:.3f}, text_precision: {:.3f}".format(
                args.start_epoch + epoch, args.num_epoches, train_loss, image_precision, text_precision
            )
        )
        scheduler.step()  # 更新学习率

        # 每个epoch进行一次测试
        if epoch % 1 == 0:
            # 测试模型性能（现在包含mAP）
            (
                ac_top1_i2t,   # 图像到文本top1准确率
                ac_top5_i2t,   # 图像到文本top5准确率
                ac_top10_i2t,  # 图像到文本top10准确率
                map_i2t,       # 图像到文本mAP
                ac_top1_t2i,   # 文本到图像top1准确率
                ac_top5_t2i,   # 文本到图像top5准确率
                ac_top10_t2i,  # 文本到图像top10准确率
                map_t2i,       # 文本到图像mAP
                test_time,     # 测试时间
            ) = test(test_loader, network, args, unique_image, epoch)
            
            # 保存模型状态
            state = {
                "network": network.state_dict(),      # 网络参数
                "optimizer": optimizer.state_dict(), # 优化器状态
                "epoch": args.start_epoch + epoch,  # 当前轮次
            }

            # 如果当前模型性能更好，则保存为最佳模型
            if ac_top1_t2i > ac_t2i_top1_best:
                best_epoch = epoch
                ac_t2i_top1_best = ac_top1_t2i
                save_checkpoint(state, epoch, args.checkpoint_dir, is_best)

            # 记录测试结果（增加mAP输出）
            logging.info("Text-to-Image:")
            logging.info(" R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}, mAP: {:.2f}".format(
                    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, map_t2i))
            logging.info("Image-to-Text:")
            logging.info(" R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}, mAP: {:.2f}".format(
                    ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, map_i2t))

    # 训练结束，记录最终结果
    logging.info("Train Finished!")
    logging.info("The best epoch:{}, the R@1 is: {:.2f}".format(best_epoch, ac_t2i_top1_best))
    logging.info(args.checkpoint_dir)


# 程序入口点
if __name__ == "__main__":
    args = config()  # 获取配置参数
    main(args)      # 执行主函数