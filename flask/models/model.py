import torch
import torch.nn as nn
import time
from .resnet import resnet50
from .bert import Bert
import torchvision.models as models
import torch
import os
import cv2
import math
import clip as openai_clip
import copy
import torch.nn.functional as F
import pickle
import numpy as np
from .modeling import VisionTransformer, CONFIGS
from .clip import VisionTransformer_clip
from .layers import TransformerDecoder, ResidualAttention



class CrossEn(nn.Module):
    """交叉熵损失函数类"""
    def __init__(self):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        """计算交叉熵损失
        Args:
            sim_matrix: 相似度矩阵
        Returns:
            sim_loss: 交叉熵损失值
        """
        logpt = F.log_softmax(sim_matrix, dim=-1)  # 对相似度矩阵进行log_softmax
        logpt = torch.diag(logpt)  # 取对角线元素（正样本对应的概率）
        nce_loss = -logpt  # 计算负对数似然损失
        sim_loss = nce_loss.mean()  # 取平均值
        return sim_loss

def gelu(x):
    """GELU激活函数"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ViT_CLIP_custom(nn.Module):
    """自定义的CLIP视觉Transformer模型"""
    def __init__(self, model_name='ViT-B/16', device=None):
        super(ViT_CLIP_custom, self).__init__()
        self.device = device
        # 加载预训练的CLIP模型
        clip_model, _ = openai_clip.load(model_name, device=self.device)
        state_dict = clip_model.visual.state_dict()
        
        # 从预训练模型中提取视觉模型的配置参数
        vision_width = state_dict["conv1.weight"].shape[0]  # 视觉模型的宽度
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("transformer.") and k.endswith(".attn.in_proj_weight")])  # Transformer层数
        vision_patch_size = state_dict["conv1.weight"].shape[-1]  # 图像块大小
        grid_size = round((state_dict["positional_embedding"].shape[0] - 1) ** 0.5)  # 网格大小
        image_resolution = vision_patch_size * grid_size  # 图像分辨率
        vision_heads = vision_width // 64  # 注意力头数
        
        # 创建自定义的视觉Transformer
        self.clip_model = VisionTransformer_clip(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads)
        # 移除投影层权重（如果存在）
        if "proj" in state_dict:
            del state_dict["proj"]
        self.clip_model.load_state_dict(state_dict)

    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像
        Returns:
            x: 图像特征
            att_score: 注意力分数
        """
        x, att_score = self.clip_model(x)
        return x, att_score


def weights_init_kaiming(m):
    """Kaiming权重初始化函数"""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        # 线性层使用Kaiming正态分布初始化
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        # 卷积层使用Kaiming正态分布初始化
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        # 批归一化层权重初始化为1，偏置初始化为0
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    """CFine主模型类
    
    该模型实现了CFine论文中的三个核心组件：
    1. MGF (Multi-Granularity Features): 多粒度特征提取
    2. CFR (Cross-modal Feature Refinement): 跨模态特征细化
    3. FCD (Fine-grained Correspondence Discovery): 细粒度对应发现
    
    同时集成了子空间投影功能，支持15个句子的处理
    """
    def __init__(self, args):
        super(Model, self).__init__()
        # 模型配置参数,超参数
        self.image_model_name = args.img_model
        self.img_k_ratio = args.img_k_ratio   # 图像特征选择比例 (默认0.1)
        self.text_k_ratio = args.text_k_ratio # 文本特征选择比例 (默认0.2)
        self.kw = args.pos_kw                 # 词级对应发现的k值 (默认3)
        self.kp = args.pos_kp                 # 像素级对应发现的k值 (默认3)
        self.scale_cs = args.scale_cs         # 交叉相似度缩放因子 (默认0.07)
        self.scale_cd = args.scale_cd         # 对应发现缩放因子 (默认0.01)

        # 初始化图像编码器
        if args.img_model == "vit":
            # 使用Vision Transformer
            config = CONFIGS[args.model_type]
            self.image_model = VisionTransformer(config)
            self.image_model.load_from(np.load(args.pretrain_dir))
        elif args.img_model == "clip":
            # 使用CLIP视觉模型
            self.image_model = ViT_CLIP_custom()
            self.image_model.clip_model.float()
            
        # 初始化文本编码器
        if args.text_model == "bert":
            self.language_model = Bert()

        # CFR组件：跨模态特征细化模块
        '''
        self.block = ResidualAttention(num_layers=1,
                                       d_model=768,
                                       n_head=12,
                                       att_type='cross')
        '''
        # 特征瓶颈层（用于特征标准化）

        self.bottleneck_image = nn.BatchNorm1d(args.subspace_dim)
        
        
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)
        
        # 文本瓶颈层始终使用原始特征维度（因为文本特征没有投影）
        #self.bottleneck_text = nn.BatchNorm1d(args.feature_size)
        #self.bottleneck_text.bias.requires_grad_(False)
        #self.bottleneck_text.apply(weights_init_kaiming)
        self.bottleneck_text = nn.BatchNorm1d(args.subspace_dim)
        nn.init.constant_(self.bottleneck_text.weight, 1)
        nn.init.constant_(self.bottleneck_text.bias, 0)

        # 相似度归一化层
        self.similarityNorm = nn.Softmax(dim=2)

        # 子空间投影配置
        self.use_subspace = hasattr(args, 'subspace_dim') and args.subspace_dim > 0
        if self.use_subspace:
            # 首先初始化子空间相关参数
            self.num_subspaces = getattr(args, 'num_subspaces', 5)  # 5个子空间
            self.subspace_dim = args.subspace_dim
            
            # 然后创建投影器
            class SimpleProjector(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.projection = nn.Linear(input_dim, output_dim)
                    nn.init.xavier_uniform_(self.projection.weight)
                
                def forward(self, x):
                    return F.normalize(self.projection(x), p=2, dim=-1)
            
            # 在Model类的__init__中替换：
            self.text_projectors = nn.ModuleList([
                SimpleProjector(args.feature_size, args.subspace_dim) 
                for _ in range(self.num_subspaces)
            ])
            
            self.img_projectors = nn.ModuleList([
                SimpleProjector(args.feature_size, args.subspace_dim) 
                for _ in range(self.num_subspaces)
            ])
            
            # 可学习的子空间权重
            self.subspace_weights = nn.Parameter(torch.ones(self.num_subspaces) / self.num_subspaces)

    def compute_subspace_features_and_similarity(self, images, tokens_all, segments_all, input_masks_all):
        """计算子空间特征和相似度（加权版本）
        
        Args:
            images: 输入图像 [B, 3, H, W]
            tokens_all: 所有句子的tokens [B, 5, seq_len]
            segments_all: 所有句子的segments [B, 5, seq_len]
            input_masks_all: 所有句子的input_masks [B, 5, seq_len]
            
        Returns:
            加权后的 (img_output, text_output, img_f, text_f, sim_cs, sim_cd, subspace_features)
        """
        
        #print(tokens_all)
        batch_size = tokens_all.size(0)
        num_sentences = tokens_all.size(1)  # 应该是5
        
        # ==================== 预先提取图像特征 ====================
        if self.image_model_name == 'clip':
            image_feats, _ = self.image_model(images)  # [B, 512]
        else:
            image_feats = self.image_model(images)[0]  # [B, N+1, D]
        img_global_original = image_feats[:, 0, :]  # [B, D] - 提取全局特征
        
        # ==================== 检测并过滤"0"句子 ====================
        # 创建有效句子的掩码，用于标识哪些句子不是"0"
        valid_sentence_mask = torch.ones(batch_size, num_sentences, dtype=torch.bool, device=tokens_all.device)
        
        # 检查每个句子是否为"0"（处理后的形式：[101, 1014, 102, 0, 0, ...]）
        # 其中101是[CLS]，1014是"0"，102是[SEP]
        for b in range(batch_size):
            for s in range(num_sentences):
                tokens = tokens_all[b, s, :]
                # 检查是否为"0"句子的模式：[CLS] 0 [SEP] [PAD]...
                # 通常"0"被tokenize后，第二个token（索引1）会是"0"对应的token ID
                if tokens[1] == 1014:  # 1014是"0"在BERT词汇表中的ID
                    valid_sentence_mask[b, s] = False
        
        # ==================== 批量文本特征提取 ====================
        # 将所有句子重塑为批量输入：[B*5, seq_len]
        tokens_batch = tokens_all.view(-1, tokens_all.size(-1))  # [B*5, seq_len]
        segments_batch = segments_all.view(-1, segments_all.size(-1))  # [B*5, seq_len]
        input_masks_batch = input_masks_all.view(-1, input_masks_all.size(-1))  # [B*5, seq_len]
        
        # 一次性处理所有句子
        text_feats_batch, _ = self.language_model(tokens_batch, segments_batch, input_masks_batch)  # [B*5, L, D]
        text_global_batch = text_feats_batch[:, 0, :]  # [B*5, D] - 提取全局特征
        
        # 重新分离每个子空间的文本特征：[B*5, D] -> [B, 5, D]
        text_global_subspaces = text_global_batch.view(batch_size, num_sentences, -1)  # [B, 5, D]
        
        # 添加调试信息
        print(f"Debug: batch_size={batch_size}, num_sentences={num_sentences}")
        print(f"Debug: text_global_batch.shape={text_global_batch.shape}")
        print(f"Debug: text_global_subspaces.shape={text_global_subspaces.shape}")
        print(f"Debug: self.num_subspaces={self.num_subspaces}")
        
        # 存储每个子空间的输出和投影特征
        subspace_outputs = []
        image_subspace_features = []
        text_subspace_features = []
        
        # ==================== 子空间处理 ====================
        for k in range(self.num_subspaces):
            print(f"Debug: Processing subspace k={k}")
            # 获取第k个子空间的文本特征
            text_global_original = text_global_subspaces[:, k, :]  # [B, D]
            
            # ==================== 过滤"0"句子 ====================
            # 对于无效的句子（"0"句子），将其特征置零
            valid_mask = valid_sentence_mask[:, k].unsqueeze(1)  # [B, 1]
            text_global_original = text_global_original * valid_mask.float()  # 将无效句子的特征置零
            
            # 在第k个子空间中进行前向传播（不再需要重复调用BERT）
            subspace_output = self._forward_single_subspace_batch_optimized(
                img_global_original, text_global_original, k
            )
            
            # 解包输出，包含投影特征
            img_output, text_output, img_f, text_f, sim_cs, sim_cd, img_projected, text_projected = subspace_output
            
            # 对投影特征也应用掩码
            text_projected = text_projected * valid_mask.float()
            
            # 保存原始输出（用于加权求和）
            subspace_outputs.append((img_output, text_output, img_f, text_f, sim_cs, sim_cd))
            
            # 保存投影特征（用于MultiSubLoss）
            image_subspace_features.append(img_projected)
            text_subspace_features.append(text_projected)
    
        # ==================== 加权求和阶段 ====================
        # 使用可学习权重进行加权求和
        weights = F.softmax(self.subspace_weights, dim=0)  # 归一化权重
    
        # 初始化加权输出
        weighted_img_output = None
        weighted_text_output = None
        weighted_img_f = None
        weighted_text_f = None
        weighted_sim_cs = None
        weighted_sim_cd = None
        
        for k, output in enumerate(subspace_outputs):
            img_output, text_output, img_f, text_f, sim_cs, sim_cd = output
            weight = weights[k]
            
            if weighted_img_output is None:
                # 初始化
                weighted_img_output = tuple(weight * feat for feat in img_output)
                weighted_text_output = tuple(weight * feat for feat in text_output)
                weighted_img_f = weight * img_f
                weighted_text_f = weight * text_f
                weighted_sim_cs = weight * sim_cs
                weighted_sim_cd = weight * sim_cd
            else:
                # 累加
                weighted_img_output = tuple(
                    w_feat + weight * feat 
                    for w_feat, feat in zip(weighted_img_output, img_output)
                )
                weighted_text_output = tuple(
                    w_feat + weight * feat 
                    for w_feat, feat in zip(weighted_text_output, text_output)
                )
                weighted_img_f += weight * img_f
                weighted_text_f += weight * text_f
                weighted_sim_cs += weight * sim_cs
                weighted_sim_cd += weight * sim_cd
    
        # 准备子空间特征信息
        subspace_features = {
            'image_subspace_features': image_subspace_features,
            'text_subspace_features': text_subspace_features,
            'projection_matrices': [self.img_projectors[k].projection.weight for k in range(self.num_subspaces)]
        }
        
        return weighted_img_output, weighted_text_output, weighted_img_f, weighted_text_f, weighted_sim_cs, weighted_sim_cd, subspace_features

    def _forward_single_subspace_batch_optimized(self, img_global_original, text_global_original, subspace_idx):
        """批量优化版本：使用预提取的图像和文本特征进行子空间前向传播
    
        Args:
            img_global_original: 预提取的图像全局特征 [B, D]
            text_global_original: 预提取的文本全局特征 [B, D]
            subspace_idx: 子空间索引
        
        Returns:
            (img_output, text_output, img_f, text_f, sim_cs, sim_cd, img_global_projected, text_global_projected)
        """
        # ==================== 子空间投影 ====================
        # 对预提取的图像特征进行子空间投影
        img_global_projected = self.img_projectors[subspace_idx](img_global_original)  # [B, subspace_dim]
    
        # 对预提取的文本特征进行子空间投影
        text_global_projected = self.text_projectors[subspace_idx](text_global_original)  # [B, subspace_dim]
    
        # 应用瓶颈层
        img_global = self.bottleneck_image(img_global_projected)
        text_global = self.bottleneck_text(text_global_projected)
    
        # ==================== 简化的相似度计算 ====================
        if self.training:
            # 获取全局特征
            img_global_feat = img_global_projected  # [B, subspace_dim]
            text_global_feat = text_global_projected  # [B, subspace_dim]
        
            # L2归一化（使用优化的F.normalize）
            img_global_norm = F.normalize(img_global_feat, p=2, dim=-1)
            text_global_norm = F.normalize(text_global_feat, p=2, dim=-1)
        
            # 计算余弦相似度矩阵
            similarity_matrix = torch.matmul(img_global_norm, text_global_norm.t())  # [B, B]
        
            # 将相似度分别设置为总相似度的一半
            sim_cs = similarity_matrix / 2.0
            sim_cd = similarity_matrix / 2.0
        else:
            # 推理模式下跳过相似度计算
            batch_size = img_global_original.size(0)
            device = img_global_original.device
            sim_cs = torch.zeros(batch_size, batch_size, device=device)
            sim_cd = torch.zeros(batch_size, batch_size, device=device)
            
        # ==================== 输出特征组装 ====================
        # 只使用全局特征
        img_output = (img_global,)
        text_output = (text_global,)
        
        # 堆叠所有特征
        img_f = torch.stack(img_output, dim=1)   # [B, 1, D]
        text_f = torch.stack(text_output, dim=1) # [B, 1, D]
    
        # 返回投影特征用于MultiSubLoss
        return img_output, text_output, img_f, text_f, sim_cs, sim_cd, img_global_projected, text_global_projected
    # ==================== 文本特征提取 ====================
    def forward(self, images, tokens_all, segments_all, input_masks_all):
        """模型前向传播
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]
            tokens_all: 所有句子的tokens [batch_size, 15, seq_len] 或 [batch_size, seq_len]
            segments_all: 所有句子的segments [batch_size, 15, seq_len] 或 [batch_size, seq_len]
            input_masks_all: 所有句子的input_masks [batch_size, 15, seq_len] 或 [batch_size, seq_len]
            
        Returns:
            训练时: (img_output, text_output, img_f, text_f, sim_cs, sim_cd, subspace_features)
            推理时: (img_output, text_output)
        """
        # 检查输入维度，确定是否使用子空间
        if self.training:
            # 训练模式：使用子空间处理并返回完整信息
            return self.compute_subspace_features_and_similarity(
                images, tokens_all, segments_all, input_masks_all
            )
        else:
            # 推理模式：仍然使用子空间处理，但只返回必要的特征
            weighted_img_output, weighted_text_output, weighted_img_f, weighted_text_f, weighted_sim_cs, weighted_sim_cd, subspace_features = self.compute_subspace_features_and_similarity(
                images, tokens_all, segments_all, input_masks_all
            )
            return weighted_img_output, weighted_text_output, subspace_features
        '''
        推理后面再考虑
        else:
            # 传统处理：[B, seq_len] 或推理模式
            if len(tokens_all.shape) == 3:
                # 如果是3维但不使用子空间，取第一个句子
                tokens = tokens_all[:, 0, :]
                segments = segments_all[:, 0, :]
                input_masks = input_masks_all[:, 0, :]
            else:
                tokens = tokens_all
                segments = segments_all
                input_masks = input_masks_all
            
            # 使用第0个子空间进行处理（或不使用子空间）
            subspace_idx = 0 if self.use_subspace else None
            
            if self.training:
                if self.use_subspace:
                    return self._forward_single_subspace(images, tokens, segments, input_masks, subspace_idx)
                else:
                    # 原始的单句子训练逻辑
                    return self._forward_single_subspace(images, tokens, segments, input_masks, 0)
            else:
                # 推理模式：只返回特征
                if self.use_subspace:
                    img_output, text_output, _, _, _, _ = self._forward_single_subspace(
                        images, tokens, segments, input_masks, subspace_idx
                    )
                else:
                    img_output, text_output, _, _, _, _ = self._forward_single_subspace(
                        images, tokens, segments, input_masks, 0
                    )
                return img_output, text_output
            '''