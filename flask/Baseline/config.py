import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging

from datasets.pedes import CuhkPedes
from models.model import Model
from utils import directory

# 配置日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def custom_collate_fn(batch):
    """
    自定义批处理函数，确保 captions 的维度正确
    
    参数:
        batch: 批次数据列表，每个元素是 (image, captions, label)
    
    返回:
        images: 图像张量 [batch_size, 3, 224, 224]
        captions: 重新组织的 captions 列表 [batch_size, 15]
        labels: 标签张量 [batch_size]
    """
    images, captions, labels = zip(*batch)
    
    # 将图像堆叠成张量
    images = torch.stack(images, 0)
    
    # 将标签转换为张量
    labels = torch.tensor(labels)
    
    # 重新组织 captions，确保形状为 [batch_size, 15]
    batch_captions = []
    for caption_list in captions:
        batch_captions.append(caption_list)
    
    return images, batch_captions, labels

def data_config(image_dir, anno_dir, batch_size, split, max_length, transform, vocab_path='', min_word_count=0, cap_transform=None):
    '''
    数据加载配置函数
    参数:
        image_dir: 图像目录路径
        anno_dir: 标注文件目录路径
        batch_size: 批处理大小
        split: 数据集划分类型 ('train', 'val', 'test')
        max_length: 文本最大长度
        transform: 图像变换函数
        vocab_path: 词汇表路径
        min_word_count: 最小词频
        cap_transform: 文本变换函数
    返回:
        loader: 数据加载器
    '''
    
    data_split = CuhkPedes(image_dir, anno_dir, split, max_length, transform, \
                            vocab_path=vocab_path, min_word_count=min_word_count, cap_transform=cap_transform)
    if split == 'train':
        shuffle = True
        loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=4, drop_last=True, collate_fn=custom_collate_fn)
    else:
        shuffle = False
        loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=4, drop_last=False, collate_fn=custom_collate_fn)
    return loader

def get_image_unique(image_dir, anno_dir, batch_size, split, max_length, transform):
    '''
    获取唯一图像列表
    参数与data_config类似
    返回:
        unique: 唯一图像列表
    '''
    return CuhkPedes(image_dir, anno_dir, split, max_length, transform).unique

def network_config(args, split='train', param=None, resume=False, model_path=None, param2=None):
    '''
    网络配置函数
    参数:
        args: 参数对象
        split: 数据集划分类型 ('train', 'test')
        param: 额外参数
        resume: 是否从检查点恢复
        model_path: 模型路径
        param2: 额外参数2
    返回:
        network: 配置好的网络模型
        optimizer: 优化器(测试模式下为None)
    '''
    # 初始化模型并配置为数据并行模式
    network = Model(args)
    network = nn.DataParallel(network).cuda()
    cudnn.benchmark = True
    args.start_epoch = 0

    # 处理网络参数
    if resume:
        # 从检查点恢复训练
        directory.check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch'] + 1
        network.load_state_dict(checkpoint['network'])
        print('==> Loading checkpoint "{}"'.format(model_path))
    else:
        # 加载预训练模型
        if model_path is not None:
            print('==> Loading from pretrained models')
            network_dict = network.state_dict()
            # 处理预训练模型的关键字
            cnn_pretrained = torch.load(model_path)
            network_keys = network_dict.keys()
            prefix = 'module.image_model.'
            update_pretrained_dict = {}
            for k,v in cnn_pretrained.items():
                # 匹配主干网络参数
                if prefix+k in network_keys:
                    update_pretrained_dict[prefix+k] = v
                # 匹配分支2网络参数
                if prefix+'branch2_'+k in network_keys:
                    update_pretrained_dict[prefix+'branch2_'+k] = v
                # 匹配分支3网络参数
                if prefix+'branch3_'+k in network_keys:
                    update_pretrained_dict[prefix+'branch3_'+k] = v
                # 提示未加载的参数
                if prefix+k not in network_keys and prefix+'branch2_'+k not in network_keys and prefix+'branch3_'+k not in network_keys:
                    print("warning: " + k + ' not load')
            network_dict.update(update_pretrained_dict)
            network.load_state_dict(network_dict)
                    
           
    # 处理优化器参数
    if split == 'test':
        # 测试模式不需要优化器
        optimizer = None
    else:
        # 为不同部分设置不同的学习参数
        # 获取语言模型参数ID
        lang_params = list(map(id, network.module.language_model.parameters()))
        # 获取图像模型参数ID
        cnn_params = list(map(id, network.module.image_model.parameters()))
        cnn_params = cnn_params + lang_params
        # 获取其他参数
        other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
        other_params = list(other_params)

        # 添加额外参数
        if param is not None:
            other_params.extend(list(param))

        if param2 is not None:
            other_params.extend(list(param2))

        # 设置参数组，为不同组设置不同的学习率和权重衰减
        param_groups = [{'params':other_params},
            {'params':network.module.image_model.parameters(), 'weight_decay':args.wd, 'lr':args.lr/10},
            {'params':network.module.language_model.parameters(), 'lr':args.lr/10}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        # 如果是恢复训练，加载优化器状态
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
    # 记录模型总参数量
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))

    return network, optimizer


def log_config(args, ca):
    '''
    日志配置函数
    参数:
        args: 参数对象
        ca: 日志文件名前缀
    '''
    filename = args.checkpoint_dir +'/' + ca + '.txt'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    # 记录所有参数
    for k in args.__dict__:
        logger.info(str(k)+' : '+str(args.__dict__[k]))


def dir_config(args):
    '''
    目录配置函数
    参数:
        args: 参数对象，包含image_dir, anno_dir, checkpoint_dir等路径
    '''
    # 检查数据集目录是否存在
    if not os.path.exists(args.image_dir):
        raise ValueError('Supply the dataset directory with --image_dir')
    if not os.path.exists(args.anno_dir):
        raise ValueError('Supply the anno file with --anno_dir')
    # 创建检查点保存目录
    directory.makedir(args.checkpoint_dir)


def lr_scheduler(optimizer, args):
    '''
    学习率调度器配置函数
    参数:
        optimizer: 优化器
        args: 参数对象，包含epoches_decay
    返回:
        scheduler: 学习率调度器
    '''
    if '_' in args.epoches_decay:
        # 多步学习率衰减
        epoches_list = args.epoches_decay.split('_')
        epoches_list = [int(e) for e in epoches_list]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list)
    else:
        # 单步学习率衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay))
    return scheduler
