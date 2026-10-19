# 导入必要的库
import torch.utils.data as data  # 导入PyTorch数据处理模块
import numpy as np  # 导入numpy用于数组操作
import os  # 导入os模块用于文件路径操作
import pickle  # 导入pickle用于序列化和反序列化Python对象
# import h5py  # h5py库用于处理HDF5格式文件（已注释）
import json  # 导入json模块用于处理JSON格式数据
from PIL import Image  # 导入PIL库中的Image模块用于图像处理
from utils.directory import check_exists  # 导入自定义函数用于检查目录是否存在
# from scipy.misc import imread, imresize  # 已注释的scipy图像处理函数
from PIL import Image  # 重复导入Image模块
import numpy as np  # 重复导入numpy
import datasets.preprocess as preprocess  # 导入自定义的预处理模块

class Namespace:
    """简单的命名空间类，用于创建具有属性的对象"""
    def __init__(self, **kwargs):
        """初始化函数，将关键字参数转换为对象属性"""
        self.__dict__.update(kwargs)

class CuhkPedes(data.Dataset):
    '''
    CUHK-PEDES数据集的PyTorch数据集类
    
    参数:
        root (string): 数据集的基础根目录，其中存在[split].pkl和[split].h5文件
        split (string): 'train', 'val'或'test'，表示数据集的划分
        transform (callable, optional): 一个函数/转换，接收PIL图像并返回转换后的向量
            例如，'transform.RandomCrop'
        target_transform (callable, optional): 一个函数/转换，接收目标并进行转换
    '''

    def __init__(self, image_root, anno_root, split, max_length, transform=None, target_transform=None, \
                 cap_transform=None, vocab_path='', min_word_count=0):
        """初始化CuhkPedes数据集
        
        参数:
            image_root: 图像根目录路径
            anno_root: 注释文件根目录路径
            split: 数据集划分（'train', 'val', 'test'）
            max_length: 文本描述的最大长度
            transform: 图像转换函数
            target_transform: 标签转换函数
            cap_transform: 描述文本转换函数
            vocab_path: 词汇表路径
            min_word_count: 最小词频阈值
        """

        self.image_root = image_root  # 图像根目录
        self.anno_root = anno_root  # 注释文件根目录
        self.max_length = max_length  # 文本描述的最大长度
        self.transform = transform  # 图像转换函数
        self.target_transform = target_transform  # 标签转换函数
        self.cap_transform = cap_transform  # 描述文本转换函数
        self.split = split.lower()  # 数据集划分（转为小写）
        self.vocab_path = vocab_path  # 词汇表路径
        self.min_word_count = min_word_count  # 最小词频阈值

        # 检查图像根目录是否存在
        if not check_exists(self.image_root):
            raise RuntimeError('数据集未找到或已损坏。' +
                               '请按照指示生成数据集')

        # 从JSON文件获取数据并读取
        data = self.get_data_from_json()
        self.read_data(data)

    
    def add_caption_to_data(self, split_data, data):
        """将描述文本添加到数据中
        
        参数:
            split_data: 划分后的数据
            data: 需要添加描述的数据，即只有id没有caption的数据
            
        返回:
            添加了描述文本的数据
        """

        # 创建文件路径到数据项的映射字典
        fname_dict = {jj['file_path']:jj for jj in split_data}
        # 创建文件路径到计数器的映射字典，初始值为0
        judge = {jj['file_path']: 0 for jj in split_data}

        # 统计每个文件路径在数据中出现的次数
        for fname in data['images_path']:
            judge[fname] = judge[fname] + 1
        
        # 为每个图像添加对应的描述文本
        caption = []
        for fname in data['images_path']:
            jj = fname_dict[fname]
            caption.append(jj['captions'][:15])

            judge[fname] = judge[fname] - 1
            
        # 确保描述文本数量与图像路径数量一致
        assert len(caption) == len(data['images_path'])
        # 将描述文本添加到数据中
        data['captions'] = caption
        return data

    
    def get_data_from_json(self):
        """从JSON文件获取数据
        
        返回:
            处理后的数据集
        """
        # 创建参数命名空间
        args = Namespace(min_word_count=self.min_word_count, remove_stopwords = None, out_root=None)

        # 加载指定划分的数据
        split_data = self.load_split(self.split)
       
        # 构建或加载词汇表
        if self.vocab_path == '':
            # 如果未指定词汇表路径，则构建新的词汇表
            vocab = preprocess.build_vocab(split_data, args, write=False)
        else:
            # 否则加载已有的词汇表
            vocab = self.load_vocab(self.vocab_path)

       
        # 处理元数据
        split_metadata = preprocess.process_metadata(self.split, split_data, args, write=False)
        # 处理解码数据
        split_decodedata = preprocess.process_decodedata(split_metadata, vocab)
        # 处理数据集
        data = preprocess.process_dataset(self.split, split_decodedata, args, write=False)
        
        # 添加描述文本到数据中
        data = self.add_caption_to_data(split_data, data)
        return data

    def load_split(self, split):   
        """加载指定划分的数据
        
        参数:
            split: 数据集划分名称
            
        返回:
            加载的JSON数据
        """
        # 构建JSON文件路径
        split_root = os.path.join(self.anno_root, split + '_reid.json')
        # 打开并加载JSON文件
        with open(split_root, 'r') as f:
            split_data = json.load(f)
        
        return split_data

    def load_vocab(self, vocab_path):    
        """加载词汇表
        
        参数:
            vocab_path: 词汇表文件路径
            
        返回:
            加载的词汇表对象
        """
        # 打开并加载pickle格式的词汇表文件
        with open(os.path.join(vocab_path), 'rb') as f:
            word_to_idx = pickle.load(f)

        # 创建词汇表对象
        vocab = preprocess.Vocabulary(word_to_idx, len(word_to_idx))
        return vocab   

    def read_data(self, data):
        """读取数据并根据划分存储到相应的属性中
        
        参数:
            data: 需要读取的数据
        """
        
        if self.split == 'train':
            # 训练集数据
            self.train_labels = data['labels']
            self.train_captions = data['captions']         
            self.train_images = data['images_path']

        elif self.split == 'val':
            # 验证集数据
            self.val_labels = data['labels']
            self.val_captions = data['captions']
            self.val_images = data['images_path']

        elif self.split == 'test':
            # 测试集数据
            self.test_labels = data['labels']
            self.test_captions = data['captions']
            self.test_images = data['images_path']

            # 创建唯一图像标识列表
            unique = []
            new_test_images = []
            for test_image in self.test_images:
                if test_image in new_test_images:
                    # 如果图像已经在新列表中，标记为非唯一(0)
                    unique.append(0)
                else:
                    # 否则标记为唯一(1)，并添加到新列表中
                    unique.append(1)
                    new_test_images.append(test_image)
            # 存储唯一图像标识列表
            self.unique = unique

        
        else:
            # 如果划分名称不是'train'、'val'或'test'，则抛出异常
            raise RuntimeError('错误的划分名称，应为"train"、"val"或"test"之一')


    def __getitem__(self, index):
        """获取指定索引的数据项
        
        参数:
              index(int): 索引
        返回:
              tuple: (图像, 标签, 描述文本)
        """
        if self.split == 'train':
            # 训练集数据
            img_path, captions, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            # 验证集数据
            img_path, captions, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            # 测试集数据
            img_path, captions, label = self.test_images[index], self.test_captions[index], self.test_labels[index]


        # 构建完整的图像路径
        img_path=os.path.join(self.image_root,img_path) 

        # 加载并处理图像
        img = Image.open(img_path)
        img = np.array(img)  # 转换为numpy数组

        # 如果图像是灰度图（只有2个维度），转换为RGB图像
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        # 应用图像转换
        if self.transform is not None:
            img = self.transform(img)

        # 应用标签转换
        if self.target_transform is not None:
            label = self.target_transform(label)

        if isinstance(captions, list) and len(captions) == 5:
            return img, captions, label
        else:
            print("错误：描述文本数量必须为5")

    def get_image(self, index):
        """获取指定索引的图像数据
        
        参数:
              index(int): 索引
        返回:
              tuple: (图像, 标签, 描述文本)
        """
        if self.split == 'train':
            img_path, caption, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]


        # 构建完整的图像路径
        img_path=os.path.join(self.image_root,img_path)

        # 加载并处理图像
        img = np.array(Image.open(img_path))
        # 如果图像是灰度图（只有2个维度），转换为RGB图像
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        # 注释掉的图像和标签转换代码
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return img, caption, label
        
    def __len__(self):
        """返回数据集的长度
        
        返回:
            数据集中的样本数量
        """
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
