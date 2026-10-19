# -*- coding: utf-8 -*-
"""
跨模态检索损失函数 (Cross-modal Retrieval Loss)
实现了基于三元组的对比学习损失，用于图像-文本跨模态检索任务

@author: zifyloo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CRLoss(nn.Module):
    """
    跨模态检索损失函数类
    
    该损失函数基于三元组损失(Triplet Loss)的思想，通过最大化正样本对的相似度
    和最小化负样本对的相似度来学习跨模态的特征表示。
    
    主要特点：
    1. 支持半困难负样本挖掘(Semi-hard negative mining)
    2. 支持困难负样本挖掘(Hard negative mining)
    3. 同时考虑图像到文本和文本到图像的检索损失
    """

    def __init__(self, opt):
        """
        初始化CRLoss损失函数
        
        Args:
            opt: 配置对象，包含以下参数：
                - gpus: GPU设备编号
                - margin: 三元组损失的边界值
                - batch_size: 批次大小
        """
        super(CRLoss, self).__init__()

        # 设置计算设备
        self.device = torch.device('cuda:{}'.format(opt.gpus))
        # 为每个样本设置相同的边界值，形状为(batch_size,)
        self.margin = np.array([opt.margin]).repeat(opt.batch_size)

    def semi_hard_negative(self, loss, margin):
        """
        半困难负样本挖掘策略
        
        选择满足以下条件的负样本：
        - 损失值小于边界值但大于0 (即 0 < loss < margin)
        - 这些样本既不是太容易(loss <= 0)也不是太困难(loss >= margin)
        
        Args:
            loss: 当前anchor与所有负样本的损失值数组
            margin: 边界值
            
        Returns:
            随机选择的半困难负样本索引，如果没有找到则返回None
        """
        # 找到满足半困难条件的负样本索引
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        # 随机选择一个半困难负样本
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def hard_negative(self, loss, margin):
        """
        困难负样本挖掘策略
        
        选择损失值最大的负样本，这些样本通常是最难区分的
        
        Args:
            loss: 当前anchor与所有负样本的损失值数组
            margin: 边界值
            
        Returns:
            损失值最大的负样本索引，如果没有找到则返回None
        """
        # 找到损失值大于边界值的负样本索引
        negative_index = np.where(np.logical_and(loss > margin, loss > 0))[0]
        # 选择损失值最大的困难负样本
        return np.argmax(loss) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels, margin, semi):
        """
        构建三元组(anchor, positive, negative)
        
        对于每个样本作为anchor，选择相同类别的样本作为positive（这里是自身），
        然后根据挖掘策略选择negative样本
        
        Args:
            similarity: 相似度矩阵，形状为(batch_size, batch_size)
            labels: 样本标签，形状为(batch_size,)
            margin: 边界值数组，形状为(batch_size,)
            semi: 是否使用半困难负样本挖掘策略
            
        Returns:
            triplets: 三元组索引，形状为(num_triplets, 3)
            return_margin: 对应的边界值，形状为(num_triplets,)
        """
        # 将张量转换为numpy数组便于处理
        #similarity = similarity.cpu().data.numpy()
        similarity = similarity.detach().cpu().numpy()
        labels = labels.cpu().data.numpy()
        triplets = []

        # 遍历每个样本作为anchor
        for idx, label in enumerate(labels):
            # 找到所有不同类别的样本作为负样本候选
            negative = np.where(labels != label)[0]

            # anchor与positive（自身）的相似度
            ap_sim = similarity[idx, idx]

            # 计算三元组损失：sim(a,n) - sim(a,p) + margin
            # 这里希望sim(a,p) > sim(a,n) + margin
            loss = similarity[idx, negative] - ap_sim + margin[idx]

            # 根据策略选择负样本
            if semi:
                negetive_index = self.semi_hard_negative(loss, margin[idx])
            else:
                negetive_index = self.hard_negative(loss, margin[idx])

            # 如果找到合适的负样本，添加到三元组列表
            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        # 如果没有找到任何三元组，至少添加一个（使用第一个负样本）
        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        # 转换为张量
        triplets = torch.LongTensor(np.array(triplets))
        # 获取对应的边界值
        return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).to(self.device)

        return triplets, return_margin

    def calculate_loss(self, similarity, label, margin, semi):
        """
        计算跨模态检索损失
        
        分别计算图像到文本和文本到图像的检索损失，然后求和
        
        Args:
            similarity: 图像-文本相似度矩阵，形状为(batch_size, batch_size)
                       similarity[i,j]表示第i个图像与第j个文本的相似度
            label: 样本标签
            margin: 边界值
            semi: 是否使用半困难负样本挖掘
            
        Returns:
            总损失值
        """
        # 获取图像为anchor的三元组（图像检索文本）
        image_triplets, img_margin = self.get_triplets(similarity, label, margin, semi)
        # 获取文本为anchor的三元组（文本检索图像），需要转置相似度矩阵
        text_triplets, txt_margin = self.get_triplets(similarity.t(), label, margin, semi)

        # 计算图像anchor的三元组损失
        # 损失 = max(0, margin - sim(a,p) + sim(a,n))
        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]  # sim(a,p)
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])  # sim(a,n)

        # 转置相似度矩阵用于文本anchor的计算
        similarity = similarity.t()
        # 计算文本anchor的三元组损失
        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]   # sim(a,p)
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])   # sim(a,n)

        # 总损失 = 图像检索损失 + 文本检索损失
        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss

    def forward(self, similarity, labels, semi):
        """
        前向传播函数
        
        Args:
            similarity: 图像-文本相似度矩阵
            labels: 样本标签
            semi: 是否使用半困难负样本挖掘策略
            
        Returns:
            跨模态检索损失值
        """
        cr_loss = self.calculate_loss(similarity, labels, self.margin, semi)
        return cr_loss

