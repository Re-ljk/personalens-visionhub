import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SubspaceProjection(nn.Module):
    """
    可学习的属性子空间投影模块
    
    该模块将图像特征投影到K个不同的子空间，每个子空间代表一种属性或语义概念。
    通过可学习的投影矩阵和权重，实现多子空间内的相似度计算和加权融合。
    
    注意：本实现只对图像特征进行投影，文本特征保持不变。
    """
    def __init__(self, feature_dim, subspace_dim, num_subspaces, use_temperature=True, init_temp=0.07):
        """
        初始化子空间投影模块
        
        Args:
            feature_dim: 输入特征维度
            subspace_dim: 每个子空间的维度
            num_subspaces: 子空间数量K
            use_temperature: 是否使用温度参数缩放相似度
            init_temp: 初始温度参数值
        """
        super(SubspaceProjection, self).__init__()
        
        self.feature_dim = feature_dim
        self.subspace_dim = subspace_dim
        self.num_subspaces = num_subspaces
        self.use_temperature = use_temperature
        
        # 创建K个子空间的图像投影矩阵
        self.img_projectors = nn.ModuleList([
            nn.Linear(feature_dim, subspace_dim, bias=False) 
            for _ in range(num_subspaces)
        ])
        
        # 初始化可学习的子空间权重
        self.subspace_weights = nn.Parameter(torch.ones(num_subspaces) / num_subspaces)
        
        # 可学习的温度参数（用于缩放相似度）
        if use_temperature:
            self.temp = nn.Parameter(torch.ones([]) * init_temp)
        else:
            self.register_buffer('temp', torch.ones([]) * init_temp)
            
        # 初始化投影矩阵为正交矩阵
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数，使投影矩阵近似正交"""
        for i in range(self.num_subspaces):
            nn.init.orthogonal_(self.img_projectors[i].weight)
    
    def add_perturbation(self, features, std=0.1):
        """添加扰动以增强鲁棒性
        
        Args:
            features: 输入特征
            std: 扰动的标准差
            
        Returns:
            添加扰动后的特征
        """
        if self.training:
            noise = torch.randn_like(features) * std
            return features + noise
        return features
    
    def compute_subspace_similarity(self, img_features, text_features_all):
        """
        计算子空间相似度
        
        参数:
            img_features: 图像特征 [B, D]
            text_features_all: 所有句子的文本特征 [B, 15, D]
        
        返回:
            similarity: 平均相似度 [B, B]
            similarities: 每个子空间的相似度列表
            weights: 每个子空间的权重
        """
        batch_size = img_features.size(0)
        feature_dim = img_features.size(1)
        
        similarities = []
        weights = []
        
        for k in range(self.num_subspaces):
            # 第k个子空间使用第k个句子的文本特征
            text_k = text_features_all[:, k, :]  # [B, D]
            
            # L2归一化
            text_k = text_k / text_k.norm(dim=-1, keepdim=True)
            
            # 图像特征投影到第k个子空间
            img_k = torch.matmul(img_features, self.projections[k])  # [B, subspace_dim]
            img_k = img_k / img_k.norm(dim=-1, keepdim=True)
            
            # 文本特征投影到第k个子空间
            text_proj_k = torch.matmul(text_k, self.projections[k])  # [B, subspace_dim]
            text_proj_k = text_proj_k / text_proj_k.norm(dim=-1, keepdim=True)
            
            # 计算相似度矩阵
            sim_k = torch.matmul(img_k, text_proj_k.transpose(0, 1))  # [B, B]
            similarities.append(sim_k)
            
            # 计算权重（可以基于相似度的方差或其他策略）
            weight_k = torch.softmax(self.weight_params[k], dim=0)
            weights.append(weight_k)
        
        # 加权平均所有子空间的相似度
        weighted_sim = torch.zeros_like(similarities[0])
        total_weight = 0
        
        for k in range(self.num_subspaces):
            weighted_sim += similarities[k] * weights[k]
            total_weight += weights[k]
        
        weighted_sim = weighted_sim / total_weight
        
        return weighted_sim, similarities, weights
    
    def forward(self, img_features, text_features_all):
        """
        前向传播
        
        参数:
            img_features: 图像特征 [B, D]
            text_features_all: 所有句子的文本特征 [B, 15, D]
        
        返回:
            similarity: 平均相似度
            similarities: 每个子空间的相似度列表
            weights: 每个子空间的权重
        """
        return self.compute_subspace_similarity(img_features, text_features_all)