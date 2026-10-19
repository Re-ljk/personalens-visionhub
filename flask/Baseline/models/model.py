import torch.nn as nn
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
        self.block = ResidualAttention(num_layers=1,
                                       d_model=768,
                                       n_head=12,
                                       att_type='cross')

        # 特征瓶颈层（用于特征标准化）
        self.bottleneck_image = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)
        self.bottleneck_text = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)
        
        # 相似度归一化层
        self.similarityNorm = nn.Softmax(dim=2)

    def forward(self, images, tokens, segments, input_masks):
        """模型前向传播
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]
            tokens: 文本token [batch_size, seq_len]
            segments: 文本段落标识 [batch_size, seq_len]
            input_masks: 文本掩码 [batch_size, seq_len]
            
        Returns:
            训练时: (img_output, text_output, img_f, text_f, sim_cs, sim_cd)
            推理时: (img_output, text_output)
        """
        # ==================== 特征提取阶段 ====================
        # 图像特征提取
        if self.image_model_name == 'clip':
            image_feats, image_att_scores = self.image_model(images)  # [B, N+1, D], [B, layers, N+1, N+1]
        else:
            image_feats = self.image_model(images)[0]  # [B, N+1, D]
            
        # 文本特征提取
        text_feats, text_att_scores = self.language_model(tokens, segments, input_masks)  # [B, L, D], [B, layers, L, L]
        
        # ==================== MGF: 多粒度特征提取 ====================
        # 全局特征提取（CLS token）
        img_global = self.bottleneck_image(image_feats[:, 0, :])    # 图像全局特征
        text_global = self.bottleneck_text(text_feats[:, 0, :])     # 文本全局特征

        # 创建文本掩码（padding位置为True）
        text_masks = torch.zeros_like(tokens).masked_fill_(tokens == 0, 1).bool()

        # ==================== 细粒度特征选择 ====================
        image_parts = []
        text_parts = []
        
        # 获取注意力分数（最后一层的CLS token对其他token的注意力）
        img_score = image_att_scores[-1][:, 0, :]   # [B, N+1]
        text_score = text_att_scores[-1][:, 0, :]   # [B, L]
        
        # 创建掩码，忽略CLS token的注意力分数
        temp_img = torch.zeros(img_score.size(0), 1)
        temp_text = torch.zeros(text_score.size(0), 1)
        score_mask_img = (torch.cat((temp_img, torch.ones(img_score.size(0), img_score.size(1)-1)), dim=1)).to(img_score.device)
        score_mask_text = (torch.cat((temp_text, torch.ones(text_score.size(0), text_score.size(1) - 1)), dim=1)).to(text_score.device)
        img_score = img_score * score_mask_img
        text_score = text_score * score_mask_text

        # Token选择
        # 图像特征选择
        image_feats_selected1 = []
        image_feats_selected2 = []
        image_feats_selected_all = []
        img_K = int(image_feats.size(1) * self.img_k_ratio)  # 选择的图像patch数量
        
        for b in range(image_feats.size(0)):
            # 根据注意力分数排序，选择最重要的patches
            _, idx = img_score[b].topk(img_score.size(1), largest=True, sorted=True)
            neg_idx1 = idx[:img_K]          # 第一组重要的patches
            neg_idx2 = idx[img_K:img_K*2]   # 第二组重要的patches
            image_feats_selected1.append(image_feats[b][neg_idx1, :])
            image_feats_selected2.append(image_feats[b][neg_idx2, :])
            image_feats_selected_all.append(image_feats[b][idx[:img_K*2], :])
            
        # 堆叠并添加CLS token
        image_feats_selected1 = torch.stack(image_feats_selected1, dim=0)
        # 通过这一步添加CLS token:image_feats[:, 0, :].unsqueeze(1)
        image_feats_selected1 = torch.cat((image_feats[:, 0, :].unsqueeze(1), image_feats_selected1), dim=1)
        image_feats_selected2 = torch.stack(image_feats_selected2, dim=0)
        image_feats_selected2 = torch.cat((image_feats[:, 0, :].unsqueeze(1), image_feats_selected2), dim=1)
        image_feats_selected_all = torch.stack(image_feats_selected_all, dim=0)

       
        # 使用ResidualAttention进行特征细化
        image_part1 = self.block(image_feats_selected1, image_feats)
        image_part2 = self.block(image_feats_selected2, image_feats)

        image_parts.append(image_part1)
        image_parts.append(image_part2)

        # 文本特征选择（类似图像特征选择过程）
        text_feats_selected1 = []
        text_mask_selected1 = []
        text_feats_selected2 = []
        text_mask_selected2 = []
        text_feats_selected_all = []
        text_K = int(text_feats.size(1) * self.text_k_ratio)  # 选择的文本token数量
        
        for b in range(text_feats.size(0)):
            # 根据注意力分数排序，选择最重要的tokens
            _, idx = text_score[b].topk(text_score.size(1), largest=True, sorted=True)
            neg_idx1 = idx[:text_K]          # 第一组重要的tokens
            neg_idx2 = idx[text_K:text_K*2]  # 第二组重要的tokens
            text_feats_selected1.append(text_feats[b][neg_idx1, :])
            text_mask_selected1.append(text_masks[b][neg_idx1])
            text_feats_selected2.append(text_feats[b][neg_idx2, :])
            text_mask_selected2.append(text_masks[b][neg_idx2])
            text_feats_selected_all.append(text_feats[b][idx[:text_K*2], :])
            
        # 堆叠并添加CLS token
        text_feats_selected1 = torch.stack(text_feats_selected1, dim=0)
        text_mask_selected1 = torch.stack(text_mask_selected1, dim=0)
        text_feats_selected1 = torch.cat((text_feats[:, 0, :].unsqueeze(1), text_feats_selected1), dim=1)
        text_mask_selected1 = torch.cat((text_masks[:, 0].unsqueeze(1), text_mask_selected1), dim=1)
        text_feats_selected2 = torch.stack(text_feats_selected2, dim=0)
        text_mask_selected2 = torch.stack(text_mask_selected2, dim=0)
        text_feats_selected2 = torch.cat((text_feats[:, 0, :].unsqueeze(1), text_feats_selected2), dim=1)
        text_mask_selected2 = torch.cat((text_masks[:, 0].unsqueeze(1), text_mask_selected2), dim=1)
        text_feats_selected_all = torch.stack(text_feats_selected_all, dim=0)

        # 使用ResidualAttention进行文本特征细化
        text_part1 = self.block(text_feats_selected1, text_feats, text_masks)
        text_part2 = self.block(text_feats_selected2, text_feats, text_masks)

        text_parts.append(text_part1)
        text_parts.append(text_part2)

        # ==================== 训练时的相似度计算 ====================
        if self.training:
             # ==================== CFR: 跨模态特征细化 ====================
            # ---------------------------
            # Cross-Similarity (CS): 交叉相似度计算
            # ---------------------------
            # 准备全局和局部特征
            G_img_token = image_feats[:, 0, :].unsqueeze(1)      # 全局图像特征 [B, 1, D]
            L_img_token = image_feats_selected_all               # 局部图像特征 [B, K, D]
            B = L_img_token.size(0)
            G_text_token = text_feats[:, 0, :].unsqueeze(1)      # 全局文本特征 [B, 1, D]
            L_text_token = text_feats_selected_all               # 局部文本特征 [B, K, D]

            # L2归一化
            G_img_token_norm = G_img_token / G_img_token.norm(dim=-1, keepdim=True)
            L_img_token_norm = L_img_token / L_img_token.norm(dim=-1, keepdim=True)
            G_text_token_norm = G_text_token / G_text_token.norm(dim=-1, keepdim=True)
            L_text_token_norm = L_text_token / L_text_token.norm(dim=-1, keepdim=True)

            # 计算image-word相似度
            G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)  # [B, B, 1, D]
            L_text_token_norm_r = L_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, B, K, D]

            sim_iw = torch.matmul(G_img_token_norm_l, L_text_token_norm_r.transpose(-2, -1)) / self.scale_cs  # [B, B, 1, K]
            weight_iw = F.softmax(sim_iw, dim=-1)  # 注意力权重
            sim_iw = torch.mul(sim_iw, weight_iw)  # 加权相似度
            sim_iw = torch.sum(sim_iw, dim=-1).squeeze()  # [B, B]
            

            # 计算pixel-text相似度
            L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)  # [B, B, K, D]
            G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, B, 1, D]

            sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / self.scale_cs  # [B, B, K, 1]
            weight_pt = F.softmax(sim_pt, dim=2)  # 注意力权重
            sim_pt = torch.mul(sim_pt, weight_pt)  # 加权相似度
            sim_pt = torch.sum(sim_pt, dim=2).squeeze()  # [B, B]
            

            # 交叉相似度：两种相似度的平均
            sim_cs = (sim_iw + sim_pt) / 2

            # ---------------------------
            # FCD: Fine-grained Correspondence Discovery (细粒度对应发现)
            # ---------------------------
            # 提取局部特征（去除CLS token）
            L_img_token1 = image_feats_selected1[:, 1:, :]   # [B, K, D]
            L_text_token1 = text_feats_selected1[:, 1:, :]   # [B, K, D]
            L_img_token_norm1 = L_img_token1 / L_img_token1.norm(dim=-1, keepdim=True)
            L_text_token_norm1 = L_text_token1 / L_text_token1.norm(dim=-1, keepdim=True)
            
            _b, _n, _c = L_text_token1.shape
            # 计算图像patch和文本token之间的相似度矩阵
            vidwordSim = torch.bmm(L_img_token_norm1, L_text_token_norm1.permute(0, 2, 1))  # [B, K_img, K_text]
            vidwordSim = self.similarityNorm(vidwordSim)
            
            # ---------- 发现正相关词汇 (posWord) -------
            # 对每个图像patch，找到最相似的k个文本tokens
            _, idxWord = vidwordSim.topk(self.kw, dim=2, largest=True, sorted=True)  # [B, K_img, kw]
            posWord = []
            for _batch in range(idxWord.shape[0]):
                posWord.append(L_text_token1[_batch, idxWord[_batch], :])  # [K_img, kw, D]
            posWord = torch.stack(posWord)  # [B, K_img, kw, D]
            posWord = torch.mean(posWord, dim=2)  # 对k个tokens取平均 [B, K_img, D]
            posWord_norm = posWord / posWord.norm(dim=-1, keepdim=True)
            
            # ---------- 发现正相关图像区域 (posClip) -------
            # 对每个文本token，找到最相似的k个图像patches
            _, idxVid = vidwordSim.topk(self.kp, dim=1, largest=True, sorted=True)  # [B, kp, K_text]
            posClip = []
            for _batch in range(idxVid.shape[0]):
                posClip.append(L_img_token1[_batch, idxVid[_batch], :])  # [kp, K_text, D]
            posClip = torch.stack(posClip)  # [B, kp, K_text, D]
            posClip = torch.mean(posClip, dim=1)  # 对k个patches取平均 [B, K_text, D]
            posClip_norm = posClip / posClip.norm(dim=-1, keepdim=True)

            # 计算对应发现的相似度
            # pixel-word相似度
            L_img_token_norm_l1 = L_img_token_norm1.unsqueeze(1).repeat(1, _b, 1, 1)  # [B, B, K_img, D]
            posWord_norm_r = posWord_norm.unsqueeze(0).repeat(_b, 1, 1, 1)  # [B, B, K_img, D]
            posClip_norm_l = posClip_norm.unsqueeze(1).repeat(1, _b, 1, 1)  # [B, B, K_text, D]
            L_text_token_norm_r1 = L_text_token_norm1.unsqueeze(0).repeat(_b, 1, 1, 1)  # [B, B, K_text, D]

            # 计算图像patch与发现的相关词汇的相似度
            sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / self.scale_cd
            sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)  # 取对角线元素
            sim_pw0 = torch.mean(sim_pw0, dim=2)  # [B, B]

            # 计算发现的相关图像区域与文本token的相似度
            sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / self.scale_cd
            sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)  # 取对角线元素
            sim_pw1 = torch.mean(sim_pw1, dim=2)  # [B, B]

            # 对应发现相似度：两种相似度的平均
            sim_cd = (sim_pw0 + sim_pw1) / 2

        # ==================== 输出特征组装 ====================
        # 组装最终的图像和文本输出特征
        img_output = (img_global,)   # 从全局特征开始
        text_output = (text_global,)

        # 添加细化后的局部特征
        for j in range(len(image_parts)):
            img_output = img_output + (self.bottleneck_image(image_parts[j][:, 0, :]),)
            text_output = text_output + (self.bottleneck_text(text_parts[j][:, 0, :]),)
            
        # 堆叠所有特征
        img_f = torch.stack(img_output, dim=1)   # [B, num_parts+1, D]
        text_f = torch.stack(text_output, dim=1) # [B, num_parts+1, D]
        
        # 根据训练/推理模式返回不同的输出,如果训练的话就返回以计算损失
        if self.training:
            return img_output, text_output, img_f, text_f, sim_cs, sim_cd
        else:
            return img_output, text_output
