import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSubLoss(nn.Module):
    """多子空间损失函数
    
    实现论文中的损失函数：
    L = L_global + α*L_attr + β*L_div
    
    其中：
    - L_global: 全局对比损失（传统图像-文本对比loss，原始CLIP中的loss）
    - L_attr: 属性级对比损失（在各子空间内计算对比损失，增强细粒度匹配）
    - L_div: 子空间多样性正则化（避免不同子空间学习重复特征，添加正交约束）
    """
    
    def __init__(self, temperature=0.07, alpha=1.0, beta=0.1):
        """
        Args:
            temperature (float): 温度参数τ，用于softmax计算
            alpha (float): 属性级对比损失权重α
            beta (float): 多样性正则化权重β
        """
        super(MultiSubLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
    def global_contrastive_loss(self, image_features, text_features):
        """
        全局对比损失：传统图像-文本对比loss（原始CLIP中的loss）
        
        L_global = -log(exp(v_i · t_i / τ) / Σ_{j=1}^B exp(v_i · t_j / τ))
        
        Args:
            image_features: 图像特征 [B, D]
            text_features: 文本特征 [B, D]
            
        Returns:
            global_loss: 全局对比损失
        """
        # L2归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # 创建标签（对角线为正样本）
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 计算交叉熵损失（图像到文本 + 文本到图像）
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        global_loss = (loss_i2t + loss_t2i) / 2
        return global_loss
    
    def attribute_contrastive_loss(self, image_subspace_features, text_subspace_features):
        """
        属性级对比损失：在各子空间内计算对比损失，增强细粒度匹配
        
        L_attr = (1/K) * Σ_{k=1}^K (-log(exp(v_i^k · t_i^k / τ) / Σ_{j=1}^B exp(v_i^k · t_j^k / τ)))
        
        Args:
            image_subspace_features: 图像子空间特征列表 [K个[B, D_k]]
            text_subspace_features: 文本子空间特征列表 [K个[B, D_k]]
            
        Returns:
            attr_loss: 属性级对比损失
        """
        K = len(image_subspace_features)  # 子空间数量
        total_loss = 0.0
        
        for k in range(K):
            # 获取第k个子空间的特征
            img_feat_k = image_subspace_features[k]  # [B, D_k]
            txt_feat_k = text_subspace_features[k]   # [B, D_k]
            
            # L2归一化
            img_feat_k = F.normalize(img_feat_k, dim=-1)
            txt_feat_k = F.normalize(txt_feat_k, dim=-1)
            
            # 计算相似度矩阵
            logits_k = torch.matmul(img_feat_k, txt_feat_k.t()) / self.temperature
            
            # 创建标签
            batch_size = img_feat_k.size(0)
            labels = torch.arange(batch_size, device=img_feat_k.device)
            
            # 计算交叉熵损失
            loss_i2t_k = F.cross_entropy(logits_k, labels)
            loss_t2i_k = F.cross_entropy(logits_k.t(), labels)
            
            subspace_loss = (loss_i2t_k + loss_t2i_k) / 2
            total_loss += subspace_loss
        
        attr_loss = total_loss / K
        return attr_loss
    
    def diversity_regularization(self, projection_matrices):
        """
        子空间多样性正则化：避免不同子空间学习重复特征，添加正交约束
        
        L_div = Σ_{k≠j} (U_k^T U_j)^2
        
        Args:
            projection_matrices: 投影矩阵列表 [K个[D, D_k]]
            
        Returns:
            div_loss: 多样性正则化损失
        """
        K = len(projection_matrices)
        total_loss = 0.0
        
        for k in range(K):
            for j in range(k + 1, K):  # 避免重复计算
                U_k = projection_matrices[k]  # [D, D_k]
                U_j = projection_matrices[j]  # [D, D_j]
                
                # 计算投影矩阵之间的内积
                # U_k^T @ U_j 的形状为 [D_k, D_j]
                cross_product = torch.matmul(U_k.t(), U_j)
                
                # 计算Frobenius范数的平方
                orthogonal_loss = torch.sum(cross_product ** 2)
                total_loss += orthogonal_loss
        
        div_loss = total_loss
        return div_loss
    
    def forward(self, image_global_features, text_global_features, 
                image_subspace_features, text_subspace_features, 
                projection_matrices=None):
        """
        前向传播计算总损失
        
        Args:
            image_global_features: 全局图像特征 [B, D]
            text_global_features: 全局文本特征 [B, D]
            image_subspace_features: 图像子空间特征列表 [K个[B, D_k]]
            text_subspace_features: 文本子空间特征列表 [K个[B, D_k]]
            projection_matrices: 投影矩阵列表 [K个[D, D_k]]，可选
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 1. 计算全局对比损失
        global_loss = self.global_contrastive_loss(image_global_features, text_global_features)
        
        # 2. 计算属性级对比损失
        attr_loss = self.attribute_contrastive_loss(image_subspace_features, text_subspace_features)
        
        # 3. 计算多样性正则化损失（如果提供了投影矩阵）
        div_loss = torch.tensor(0.0, device=image_global_features.device)
        if projection_matrices is not None:
            div_loss = self.diversity_regularization(projection_matrices)
        
        # 4. 计算总损失
        total_loss = global_loss + self.alpha * attr_loss + self.beta * div_loss
        
        # 返回损失字典用于监控
        loss_dict = {
            'total_loss': total_loss,
            'global_loss': global_loss,
            'attr_loss': attr_loss,
            'div_loss': div_loss
        }
        
        return total_loss, loss_dict


# 使用示例
if __name__ == "__main__":
    # 创建损失函数
    criterion = MultiSubLoss(temperature=0.07, alpha=1.0, beta=0.1)
    
    # 模拟数据
    batch_size = 32
    global_dim = 768
    subspace_dim = 64
    num_subspaces = 4
    
    # 全局特征
    img_global = torch.randn(batch_size, global_dim)
    txt_global = torch.randn(batch_size, global_dim)
    
    # 子空间特征
    img_subspaces = [torch.randn(batch_size, subspace_dim) for _ in range(num_subspaces)]
    txt_subspaces = [torch.randn(batch_size, subspace_dim) for _ in range(num_subspaces)]
    
    # 投影矩阵
    projections = [torch.randn(global_dim, subspace_dim) for _ in range(num_subspaces)]
    
    # 计算损失
    total_loss, loss_dict = criterion(
        img_global, txt_global,
        img_subspaces, txt_subspaces,
        projections
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Global Loss: {loss_dict['global_loss'].item():.4f}")
    print(f"Attribute Loss: {loss_dict['attr_loss'].item():.4f}")
    print(f"Diversity Loss: {loss_dict['div_loss'].item():.4f}")