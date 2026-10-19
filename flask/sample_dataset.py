import torch.utils.data as data
import json
import os
import numpy as np
from PIL import Image

class SampleDataset(data.Dataset):
    """用于处理5个样本数据的数据集类"""
    
    def __init__(self, sample_json_path, image_root, transform=None):
        """
        Args:
            sample_json_path: 5个样本的JSON文件路径
            image_root: 图像根目录
            transform: 图像变换
        """
        self.image_root = image_root
        self.transform = transform
        
        # 加载5个样本数据
        with open(sample_json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        # 提取图像路径、描述和标签
        self.images = [sample['file_path'] for sample in self.samples]
        self.captions = [sample['captions'] for sample in self.samples]
        self.labels = [sample['id'] for sample in self.samples]
        
        # 创建唯一图像标识（所有样本都标记为唯一）
        self.unique = [1] * len(self.samples)
        
        print(f"加载了 {len(self.samples)} 个样本")
        for i, sample in enumerate(self.samples):
            print(f"样本 {i}: {sample['file_path']}, 标签: {sample['id']}")
    
    def __getitem__(self, index):
        # 获取图像路径并拼接完整路径
        image_path = os.path.join(self.image_root, self.images[index])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 获取captions并确保包含5个元素
        captions = self.captions[index]
        
        # 确保captions列表包含5个元素
        if len(captions) < 5:
            # 如果不足5个，重复最后一个caption直到达到5个
            captions = captions + [captions[-1]] * (5 - len(captions))
        elif len(captions) > 5:
            # 如果超过5个，只取前5个
            captions = captions[:5]
        
        # 获取标签
        label = self.labels[index]
        
        return image, captions, label
    
    def __len__(self):
        return len(self.samples)