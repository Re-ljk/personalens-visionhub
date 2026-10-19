# -*- coding: utf-8 -*-
"""
损失函数和评估指标模块

该模块包含了用于跨模态人员重识别任务的损失函数和评估指标：
1. CMPM (Cross-Modal Projection Matching) 损失：跨模态投影匹配损失
2. CMPC (Cross-Modal Projection Classification) 损失：跨模态投影分类损失
3. 多样性损失：用于增强特征的多样性
4. Top-k检索准确率计算：用于评估模型性能

作者：CFine项目组
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# 配置日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EMA:
    """指数移动平均（Exponential Moving Average）类
    
    用于维护参数的指数移动平均值，常用于模型参数的平滑更新
    """
    
    def __init__(self, decay=0.999):
        """初始化EMA
        
        Args:
            decay (float): 衰减系数，控制历史值的权重，默认为0.999
        """
        self.decay = decay  # 衰减系数
        self.shadow = {}    # 存储影子参数的字典

    def register(self, name, val):
        """注册需要跟踪的参数
        
        Args:
            name (str): 参数名称
            val (torch.Tensor): 参数值
        """
        self.shadow[name] = val.cpu().detach()

    def get(self, name):
        """获取指定参数的EMA值
        
        Args:
            name (str): 参数名称
            
        Returns:
            torch.Tensor: 参数的EMA值
        """
        return self.shadow[name]

    def update(self, name, x):
        """更新指定参数的EMA值
        
        Args:
            name (str): 参数名称
            x (torch.Tensor): 新的参数值
        """
        assert name in self.shadow
        # EMA更新公式：new_average = (1-decay) * new_val + decay * old_average
        new_average = (1.0 - self.decay) * x.cpu().detach() + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()


def l2norm(x):
    """对输入张量进行L2归一化
    
    Args:
        x (torch.Tensor): 输入张量
        
    Returns:
        torch.Tensor: L2归一化后的张量
    """
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()  # 计算L2范数
    return torch.div(x, norm)  # 除以范数进行归一化


def pairwise_distance(A, B):
    """计算两个点集之间的成对欧氏距离
    
    Args:
        A (torch.Tensor): 点集A，形状为(m, n)，m个n维点
        B (torch.Tensor): 点集B，形状为(k, n)，k个n维点
        
    Returns:
        torch.Tensor: 距离矩阵，形状为(m, k)，元素(i,j)表示A[i]和B[j]之间的距离
    """
    # 计算每个点的平方和
    A_square = torch.sum(A * A, dim=1, keepdim=True)  # (m, 1)
    B_square = torch.sum(B * B, dim=1, keepdim=True)  # (k, 1)
    
    # 使用展开的欧氏距离公式：||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())
    
    return distance


def one_hot_coding(index, k):
    """将索引转换为one-hot编码
    
    Args:
        index (torch.Tensor): 类别索引
        k (int): 类别总数
        
    Returns:
        torch.Tensor: one-hot编码矩阵
    """
    if type(index) is torch.Tensor:
        length = len(index)
    else:
        length = 1
    
    # 创建零矩阵
    out = torch.zeros((length, k), dtype=torch.int64).cuda()
    index = index.reshape((len(index), 1))
    # 在对应位置填入1
    out.scatter_(1, index, 1)
    return out


# 已弃用的约束损失函数（由于内存使用过大）
def constraints_old(features, labels):
    """计算约束损失（旧版本，已弃用）
    
    该函数由于内存使用过大而被弃用
    """
    distance = pairwise_distance(features, features)
    labels_reshape = torch.reshape(labels, (features.shape[0], 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    # 计算匹配对的平均损失
    num = torch.sum(labels_mask) - features.shape[0]
    if num == 0:
        con_loss = 0.0
    else:
        con_loss = torch.sum(distance * labels_mask) / num

    return con_loss


def constraints(features, labels):
    """计算约束损失（优化版本）
    
    通过分组计算来减少内存使用
    
    Args:
        features (torch.Tensor): 特征张量
        labels (torch.Tensor): 标签张量
        
    Returns:
        float: 约束损失值
    """
    labels = torch.reshape(labels, (labels.shape[0], 1))
    con_loss = AverageMeter()  # 用于计算平均损失
    index_dict = {k.item() for k in labels}  # 获取所有唯一标签
    
    # 对每个标签分别计算约束损失
    for index in index_dict:
        labels_mask = labels == index  # 创建当前标签的掩码
        feas = torch.masked_select(features, labels_mask)  # 选择当前标签对应的特征
        feas = feas.view(-1, features.shape[1])  # 重塑特征张量
        distance = pairwise_distance(feas, feas)  # 计算同类特征之间的距离
        num = feas.shape[0] * (feas.shape[0] - 1)  # 计算配对数量
        loss = torch.sum(distance) / num  # 计算平均距离
        con_loss.update(loss, n=num / 2)  # 更新平均损失
    
    return con_loss.avg


def constraints_loss(data_loader, network, args):
    """计算整个数据集的约束损失
    
    Args:
        data_loader: 数据加载器
        network: 神经网络模型
        args: 参数配置
        
    Returns:
        tuple: (图像约束损失, 文本约束损失)
    """
    network.eval()  # 设置模型为评估模式
    max_size = args.batch_size * len(data_loader)
    
    # 初始化特征和标签存储库
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    
    index = 0
    con_images = 0.0
    con_text = 0.0
    
    # 提取所有特征
    with torch.no_grad():
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]
            
            # 前向传播获取特征
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            
            # 存储特征和标签
            images_bank[index : index + interval] = image_embeddings
            text_bank[index : index + interval] = text_embeddings
            labels_bank[index : index + interval] = labels
            index = index + interval
            
        # 截取实际使用的部分
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]

    # 根据配置计算约束损失
    if args.constraints_text:
        con_text = constraints(text_bank, labels_bank)
    if args.constraints_images:
        con_images = constraints(images_bank, labels_bank)

    return con_images, con_text


class Loss(nn.Module):
    """跨模态人员重识别损失函数类
    
    该类实现了CFine论文中的主要损失函数：
    1. CMPM (Cross-Modal Projection Matching) 损失
    2. CMPC (Cross-Modal Projection Classification) 损失
    3. 多样性损失
    """
    
    def __init__(self, args):
        """初始化损失函数
        
        Args:
            args: 包含模型配置的参数对象
        """
        super(Loss, self).__init__()
        self.CMPM = args.CMPM  # 是否使用CMPM损失
        self.CMPC = args.CMPC  # 是否使用CMPC损失
        self.epsilon = args.epsilon  # 防止除零的小常数
        self.num_classes = args.num_classes  # 类别数量（身份数量）
        
        # 初始化分类器权重矩阵W
        if args.resume:
            # 如果是恢复训练，从检查点加载权重
            checkpoint = torch.load(args.model_path)
            self.W = Parameter(checkpoint["W"])
            print("=========> Loading in parameter W from pretrained models")
        else:
            # 否则随机初始化权重矩阵，形状为(feature_size, num_classes)
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()

    def init_weight(self):
        """初始化权重矩阵W
        
        使用Xavier均匀分布初始化
        """
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def diversity_loss(self, x):
        """计算多样性损失（版本1）
        
        该损失函数鼓励不同部分的特征具有多样性，避免特征冗余
        
        Args:
            x (torch.Tensor): 输入特征，形状为(batch_size, num_parts, feature_dim)
            
        Returns:
            torch.Tensor: 多样性损失值
        """
        x = l2norm(x)  # 对特征进行L2归一化
        gram_x = x.bmm(x.transpose(1, 2))  # 计算Gram矩阵（特征相关性矩阵）
        
        # 创建单位矩阵掩码，用于屏蔽对角线元素
        I = torch.autograd.Variable(
            (torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1)
        )
        if torch.cuda.is_available():
            I = I.cuda()
        
        gram_x.masked_fill_(I, 0.0)  # 将对角线元素设为0
        
        # 计算Frobenius范数作为多样性损失
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (x.size(1) ** 2)
        return loss.mean()

    def diversity_loss2(self, features):
        """计算多样性损失（版本2，优化版本）
        
        Args:
            features (torch.Tensor): 输入特征，形状为(batch_size, num_parts, num_feat)
            
        Returns:
            torch.Tensor: 多样性损失值
        """
        batch_size, num_parts, num_feat = features.size()
        
        # 对特征进行L2归一化
        features_norm = F.normalize(features, dim=2)
        
        # 计算特征相关性矩阵
        feaures_t = features_norm.transpose(1, 2)
        loss_tentor = torch.matmul(features_norm, feaures_t)
        
        # 计算多样性损失：总相关性减去对角线元素，然后归一化
        diversity_loss = (torch.sum(loss_tentor) - 1 * batch_size * num_parts) / (
            num_parts * (num_parts - 1)
        )
        return diversity_loss

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """计算CMPC（跨模态投影分类）损失
        
        CMPC损失通过将图像特征投影到文本空间（或反之）来进行分类，
        增强跨模态特征的判别能力。
        
        Args:
            image_embeddings (torch.Tensor): 图像特征嵌入
            text_embeddings (torch.Tensor): 文本特征嵌入
            labels (torch.Tensor): 真实标签
            
        Returns:
            tuple: (CMPC损失, 图像分类精度, 文本分类精度)
        """
        criterion = nn.CrossEntropyLoss(reduction="mean")  # 交叉熵损失函数
        
        # 对分类器权重进行L2归一化
        self.W_norm = self.W / self.W.norm(dim=0)
        
        # 对图像和文本特征进行L2归一化
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # 跨模态投影：将图像特征投影到文本空间
        image_proj_text = (
            torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        )
        
        # 跨模态投影：将文本特征投影到图像空间
        text_proj_image = (
            torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm
        )

        # 计算分类logits
        image_logits = torch.matmul(image_proj_text, self.W_norm)  # 图像投影后的分类得分
        text_logits = torch.matmul(text_proj_image, self.W_norm)   # 文本投影后的分类得分

        # 计算CMPC损失：图像到文本投影的分类损失 + 文本到图像投影的分类损失
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        # 计算分类精度
        image_pred = torch.argmax(image_logits, dim=1)  # 图像分类预测
        text_pred = torch.argmax(text_logits, dim=1)    # 文本分类预测

        image_precision = torch.mean((image_pred == labels).float())  # 图像分类精度
        text_precision = torch.mean((text_pred == labels).float())    # 文本分类精度

        return cmpc_loss, image_precision, text_precision

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """计算CMPM（跨模态投影匹配）损失
        
        CMPM损失通过最小化跨模态投影后的KL散度来学习模态间的对应关系，
        使得相同身份的图像和文本特征在投影后更加相似。
        
        Args:
            image_embeddings (torch.Tensor): 图像特征嵌入
            text_embeddings (torch.Tensor): 文本特征嵌入
            labels (torch.Tensor): 真实标签
            
        Returns:
            tuple: (CMPM损失, 正样本对平均相似度, 负样本对平均相似度)
        """
        batch_size = image_embeddings.shape[0]
        
        # 创建标签匹配矩阵
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()  # 计算标签差异矩阵
        labels_mask = labels_dist == 0  # 相同标签的位置为True

        # 对特征进行L2归一化
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        
        # 计算跨模态相似度矩阵
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())  # 图像到文本的投影相似度
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())  # 文本到图像的投影相似度

        # 归一化真实匹配分布
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        # 计算预测分布（使用softmax）
        i2t_pred = F.softmax(image_proj_text, dim=1)  # 图像到文本的预测分布
        
        # 计算图像到文本的KL散度损失
        i2t_loss = i2t_pred * (
            F.log_softmax(image_proj_text, dim=1)
            - torch.log(labels_mask_norm + self.epsilon)
        )

        # 计算文本到图像的预测分布和KL散度损失
        t2i_pred = F.softmax(text_proj_image, dim=1)  # 文本到图像的预测分布
        t2i_loss = t2i_pred * (
            F.log_softmax(text_proj_image, dim=1)
            - torch.log(labels_mask_norm + self.epsilon)
        )

        # 总的CMPM损失
        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
            torch.sum(t2i_loss, dim=1)
        )

        # 计算余弦相似度矩阵
        sim_cos = torch.matmul(image_norm, text_norm.t())

        # 计算正样本对和负样本对的平均相似度
        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))      # 正样本对平均相似度
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0)) # 负样本对平均相似度

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def forward(
        self, image_embeddings, text_embeddings, img_f, text_f, labels, lambda_diversity
    ):
        """损失函数的前向传播
        
        Args:
            image_embeddings (list): 图像特征嵌入列表（多个子空间）
            text_embeddings (list): 文本特征嵌入列表（多个子空间）
            img_f (torch.Tensor): 图像的多部分特征
            text_f (torch.Tensor): 文本的多部分特征
            labels (torch.Tensor): 真实标签
            lambda_diversity (float): 多样性损失的权重系数
            
        Returns:
            tuple: (CMPM损失, CMPC损失, 总损失, 图像精度, 文本精度, 正样本相似度, 负样本相似度)
        """
        # 初始化损失和精度变量
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim = 0.0
        
        # 计算CMPM损失
        if self.CMPM:
            cmpm_loss = 0
            for i in range(len(image_embeddings)):
                cmpm_loss1, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(
                    image_embeddings[i], text_embeddings[i], labels
                )
                # 对前两个子空间给予更高权重，其他子空间权重较低
                if i == 0 or i == 1:
                    cmpm_loss += cmpm_loss1
                else:
                    cmpm_loss += cmpm_loss1 * 0.1  # 降低其他子空间的权重

        # 计算CMPC损失
        if self.CMPC:
            cmpc_loss = 0
            for i in range(len(image_embeddings)):
                cmpc_loss1, image_precision, text_precision = self.compute_cmpc_loss(
                    image_embeddings[i], text_embeddings[i], labels
                )
                cmpc_loss += cmpc_loss1  # 所有子空间的CMPC损失权重相等

        # 计算总损失：CMPM损失 + CMPC损失 + 多样性损失
        loss = cmpm_loss + cmpc_loss
        loss += lambda_diversity * (
            self.diversity_loss(img_f[:,1:]) + self.diversity_loss(text_f[:,1:])
        )
        
        return (
            cmpm_loss,
            cmpc_loss,
            loss,
            image_precision,
            text_precision,
            pos_avg_sim,
            neg_avg_sim,
        )


class AverageMeter(object):
    """计算和存储平均值和当前值的工具类
    
    从 https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262 导入
    用于跟踪训练过程中的各种指标
    """

    def __init__(self):
        """初始化AverageMeter"""
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val = 0    # 当前值
        self.avg = 0    # 平均值
        self.sum = 0    # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """更新统计量
        
        Args:
            val: 新的值
            n: 该值的权重（默认为1）
        """
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count


def compute_topk(
    query, gallery, target_query, target_gallery, k=[1, 10], reverse=False
):
    """计算查询集和图库集之间的top-k检索准确率
    
    该函数用于评估跨模态检索任务的性能，计算在top-k个最相似的候选中
    找到正确匹配的准确率。
    
    Args:
        query (torch.Tensor): 查询特征，形状为[num_heads+1, batch_size, feature_size]，通常是图像特征
        gallery (torch.Tensor): 图库特征，形状为[num_heads+1, batch_size, feature_size]，通常是文本特征
        target_query (torch.Tensor): 查询集的标签
        target_gallery (torch.Tensor): 图库集的标签
        k (list): 要计算的top-k值列表，默认为[1, 10]
        reverse (bool): 是否同时计算反向匹配（文本到图像和图像到文本），默认为False
    
    Returns:
        tuple: (包含各个top-k准确率的列表, 查询集和图库集之间的相似度矩阵)
    """
   
    result = []  # 初始化结果列表
    
    # 调整特征维度顺序：从[num_heads+1, batch_size, feature_size]到[batch_size, num_heads+1, feature_size]
    query = query.permute(1,0,2)
    gallery = gallery.permute(1,0,2)
    
    # 将多头特征展平为单一特征向量
    test_query = query.reshape(query.size(0), -1)      # [batch_size, (num_heads+1)*feature_size]
    test_gallery = gallery.reshape(gallery.size(0), -1) # [batch_size, (num_heads+1)*feature_size]
    
    # 对特征进行L2归一化
    query_norm = test_query / test_query.norm(dim=1, keepdim=True)
    gallery_norm = test_gallery / test_gallery.norm(dim=1, keepdim=True)
    
    # 计算归一化后的查询特征和图库特征之间的余弦相似度矩阵
    score = torch.matmul(query_norm, gallery_norm.t())
    
    # 计算查询到图库的top-k准确率（图像到文本检索）
    result.extend(topk(score, target_gallery, target_query, k, dim=1))
    
    # 如果reverse为True，计算图库到查询的top-k准确率（文本到图像检索）
    if reverse:
        result.extend(topk(score, target_query, target_gallery, k, dim=0))
    
    return result, score  # 返回top-k准确率列表和相似度矩阵


def topk(sim, target_gallery, target_query, k=[1, 10], dim=1):
    """计算相似度矩阵的top-k准确率
    
    Args:
        sim (torch.Tensor): 相似度矩阵，查询集和图库集之间的余弦相似度
        target_gallery (torch.Tensor): 图库集的标签
        target_query (torch.Tensor): 查询集的标签
        k (list): 要计算的top-k值列表，默认为[1, 10]
        dim (int): 计算top-k的维度，1表示行方向（查询到图库），0表示列方向（图库到查询）
    
    Returns:
        list: 包含各个top-k准确率的列表
    """
    result = []  # 初始化结果列表
    maxk = max(k)  # 获取最大的k值
    size_total = len(target_query)  # 获取查询集的总数量
    
    # 获取相似度最大的前maxk个索引
    _, pred_index = sim.topk(maxk, dim, True, True)
    
    # 根据索引获取预测的标签
    pred_labels = target_gallery[pred_index]
    
    if dim == 1:
        # 如果dim=1（行方向），转置预测标签矩阵以便于后续计算
        pred_labels = pred_labels.t()
    
    # 判断预测标签是否与真实标签相等
    correct = pred_labels.eq(target_query.view(1, -1).expand_as(pred_labels))

    # 计算各个top-k的准确率
    for topk in k:
        # 对前topk个预测结果进行求和，得到每个查询的正确匹配数
        correct_k = torch.sum(correct[:topk], dim=0)
        
        # 统计至少有一个正确匹配的查询数量
        correct_k = torch.sum(correct_k > 0).float()
        
        # 计算准确率（百分比）并添加到结果列表
        result.append(correct_k * 100 / size_total)
    
    return result  # 返回top-k准确率列表


def compute_ap(sim, target_gallery, target_query, dim=1):
    """计算Average Precision (AP)
    
    Args:
        sim (torch.Tensor): 相似度矩阵，查询集和图库集之间的余弦相似度
        target_gallery (torch.Tensor): 图库集的标签
        target_query (torch.Tensor): 查询集的标签
        dim (int): 计算方向，1表示行方向（查询到图库），0表示列方向（图库到查询）
    
    Returns:
        float: Average Precision值
    """
    # 获取排序后的索引（按相似度降序排列）
    _, pred_index = sim.sort(dim, descending=True)
    
    # 根据索引获取预测的标签
    pred_labels = target_gallery[pred_index]
    
    if dim == 1:
        # 如果dim=1（行方向），转置预测标签矩阵
        pred_labels = pred_labels.t()
    
    # 判断预测标签是否与真实标签相等
    correct = pred_labels.eq(target_query.view(1, -1).expand_as(pred_labels))
    
    ap_list = []
    num_queries = correct.size(1)
    
    # 为每个查询计算AP
    for i in range(num_queries):
        query_correct = correct[:, i].float()  # 当前查询的正确匹配情况
        
        # 计算相关项目的总数
        num_relevant = torch.sum(query_correct)
        
        if num_relevant == 0:
            ap_list.append(0.0)
            continue
        
        # 计算累积精度
        cumulative_correct = torch.cumsum(query_correct, dim=0)
        precision_at_k = cumulative_correct / torch.arange(1, len(query_correct) + 1, dtype=torch.float32, device=query_correct.device)
        
        # 只在相关项目处计算精度
        relevant_precisions = precision_at_k * query_correct
        
        # 计算AP
        ap = torch.sum(relevant_precisions) / num_relevant
        ap_list.append(ap.item())
    
    return ap_list


def compute_map(sim, target_gallery, target_query, dim=1):
    """计算mean Average Precision (mAP)
    
    Args:
        sim (torch.Tensor): 相似度矩阵，查询集和图库集之间的余弦相似度
        target_gallery (torch.Tensor): 图库集的标签
        target_query (torch.Tensor): 查询集的标签
        dim (int): 计算方向，1表示行方向（查询到图库），0表示列方向（图库到查询）
    
    Returns:
        float: mean Average Precision值（百分比）
    """
    ap_list = compute_ap(sim, target_gallery, target_query, dim)
    map_score = sum(ap_list) / len(ap_list) * 100  # 转换为百分比
    return map_score


def compute_topk_and_map(
    query, gallery, target_query, target_gallery, k=[1, 10], reverse=False
):
    """计算查询集和图库集之间的top-k检索准确率和mAP
    
    该函数扩展了原有的compute_topk函数，增加了mAP计算功能。
    
    Args:
        query (torch.Tensor): 查询特征，形状为[num_heads+1, batch_size, feature_size]
        gallery (torch.Tensor): 图库特征，形状为[num_heads+1, batch_size, feature_size]
        target_query (torch.Tensor): 查询集的标签
        target_gallery (torch.Tensor): 图库集的标签
        k (list): 要计算的top-k值列表，默认为[1, 10]
        reverse (bool): 是否同时计算反向匹配，默认为False
    
    Returns:
        tuple: (包含各个top-k准确率和mAP的列表, 查询集和图库集之间的相似度矩阵)
    """
    result = []  # 初始化结果列表
    
    # 调整特征维度顺序：从[num_heads+1, batch_size, feature_size]到[batch_size, num_heads+1, feature_size]
    query = query.permute(1,0,2)
    gallery = gallery.permute(1,0,2)
    
    # 将多头特征展平为单一特征向量
    test_query = query.reshape(query.size(0), -1)      # [batch_size, (num_heads+1)*feature_size]
    test_gallery = gallery.reshape(gallery.size(0), -1) # [batch_size, (num_heads+1)*feature_size]
    
    # 对特征进行L2归一化
    query_norm = test_query / test_query.norm(dim=1, keepdim=True)
    gallery_norm = test_gallery / test_gallery.norm(dim=1, keepdim=True)
    
    # 计算归一化后的查询特征和图库特征之间的余弦相似度矩阵
    score = torch.matmul(query_norm, gallery_norm.t())
    
    # 计算查询到图库的top-k准确率和mAP（图像到文本检索）
    result.extend(topk(score, target_gallery, target_query, k, dim=1))
    map_i2t = compute_map(score, target_gallery, target_query, dim=1)
    result.append(map_i2t)
    
    # 如果reverse为True，计算图库到查询的top-k准确率和mAP（文本到图像检索）
    if reverse:
        result.extend(topk(score, target_query, target_gallery, k, dim=0))
        map_t2i = compute_map(score, target_query, target_gallery, dim=0)
        result.append(map_t2i)
    
    return result, score  # 返回top-k准确率、mAP列表和相似度矩阵
