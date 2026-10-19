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
        """获取指定索引的数据项"""
        img_path = self.images[index]
        captions = self.captions[index]
        label = self.labels[index]
        
        # 构建完整的图像路径
        full_img_path = os.path.join(self.image_root, img_path)
        
        # 加载并处理图像
        try:
            img = Image.open(full_img_path)
            img = np.array(img)
            
            # 如果是灰度图，转换为RGB
            if len(img.shape) == 2:
                img = np.dstack((img, img, img))
            img = Image.fromarray(img)
            
            # 应用图像变换
            if self.transform is not None:
                img = self.transform(img)
                
        except Exception as e:
            print(f"无法加载图像 {full_img_path}: {e}")
            # 创建一个默认的黑色图像
            img = Image.new('RGB', (224, 224), color='black')
            if self.transform is not None:
                img = self.transform(img)
        
        return img, captions, label
    
    def __len__(self):
        return len(self.samples)