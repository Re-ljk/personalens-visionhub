import pickle
import json
import argparse
import string
import os
from zhon import hanzi
from collections import namedtuple

import nltk
"""
- 数据准备   ：读取原始JSON数据，包含图像路径、描述文本和分割信息
- 文本处理   ：将描述文本转换为标记列表，添加开始和结束标记
- 词汇表构建 ：统计词频，过滤低频词，生成词汇索引
- 数据分割   ：将数据分为训练集、验证集和测试集
- 元数据转换 ：将原始数据转换为结构化的元数据形式
- 数据解码   ：使用词汇表将文本描述转换为数字ID序列
- 数据排序   ：按图像ID对数据进行排序，创建标签范围字典
- 数据保存   ：将处理后的数据序列化为pickle文件
"""

#创建目录，如果目录不存在的话
#root - 目录路径
def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)

#- 作用：将数据写入 JSON 文件
#- 参数： data - 要写入的数据， root - 文件路径
def write_json(data, root):
    with open(root, 'w') as f:
        json.dump(data, f)
#两个元组存储基本数据信息
ImageMetaData = namedtuple('ImageMetaData', ['id', 'image_path', 'captions', 'split'])
ImageDecodeData = namedtuple('ImageDecodeData', ['id', 'image_path', 'captions_id', 'split'])

class Vocabulary(object):
    """
    Vocabulary wrapper
    """
    def __init__(self, vocab, unk_id):
        """
        :param vocab: A dictionary of word to word_id
        :param unk_id: Id of the bad/unknown words
        """
        self._vocab = vocab
        self._unk_id = unk_id
        self._reverse_vocab = {v: k for k, v in vocab.items()}# 创建反向映射字典
    #这个方法将单词转换为ID：如果单词不在词汇表中，返回未知词ID,否则，返回该单词在词汇表中对应的ID
    def word_to_id(self, word):
        if word not in self._vocab:
            return self._unk_id
        return self._vocab[word]
    #这个方法将ID转换回单词：如果ID不在反向词汇表中，返回空字符串 ,否则，返回该ID在反向词汇表中对应的单词
    def id_to_word(self, id):
        if id not in self._reverse_vocab:
            return ''
        else:
            return self._reverse_vocab[id]

#- 作用：将描述文本转换为标记（tokens）
#- 处理：移除标点符号和空白字符，将文本分解为单个字符，并添加开始和结束标记
def cap2tokens(cap):
    exclude = set(string.punctuation + string.whitespace + hanzi.punctuation)#删除所有标点（中英文）、空格
    caption = ''.join(c for c in cap if c not in exclude)
    tokens = [letter for letter in caption]#将字符串按单个字符拆分
    # print(tokens)
    tokens = add_start_end(tokens)
    return tokens

#为标记列表添加开始和结束标记
def add_start_end(tokens, start_word='<START>', end_word='<END>'):
    """
    Add start and end words for a caption
    """
    tokens_processed = [start_word]
    tokens_processed.extend(tokens)
    tokens_processed.append(end_word)
    return tokens_processed

#处理图像描述文本，将每个描述转换为标记列表
def process_captions(imgs):
    for img in imgs:
        img['processed_tokens'] = []
        for s in img['captions']:
            tokens = cap2tokens(s)
            img['processed_tokens'].append(tokens)

#统计词频，过滤低频词，生成词汇索引，保存词汇表
def build_vocab(imgs, args, write=True):
    # 初始化空字典用于统计词频
    counts = {}
    
    # 遍历所有图像数据
    for img in imgs:
        # 遍历每个图像的所有描述文本（已处理为token列表）
        for tokens in img['processed_tokens']:
            # 遍历描述中的每个单词/标记
            for word in tokens:
                # 更新词频统计：若单词存在则+1，否则初始化为1
                counts[word] = counts.get(word, 0) + 1

    # filter uncommon words and sort by descending count.
    # word_counts: a list of (words, count) for words satisfying the condition.
    # 初始化停用词列表
    stop_words = []
    
    # 如果设置了停用词文件
    if args.remove_stopwords is not None:
        # 打开停用词文件
        with open(args.remove_stopwords) as f:
            # 读取所有行
            lines = f.readlines()
        # 去除每行首尾空白字符
        stop_words = [l.strip() for l in lines]
    # 筛选满足条件的单词：(词频≥阈值)且(非停用词)
    word_counts  = [(w,n) for w,n in counts.items() if n >= args.min_word_count and w not in stop_words]
    
    # 按词频降序排序
    word_counts.sort(key = lambda x : x[1], reverse=True)

    # 找出被过滤的单词（低频词或停用词）
    words_out = [(w,n) for w,n in counts.items() 
                 if n < args.min_word_count or w in stop_words]
    
    # 计算被过滤的单词数量和总出现次数
    bad_words = len(words_out)
    bad_count = len([x[1] for x in words_out])

    # save the word counts file
    # 保存词汇统计信息
    if write:
        # 保存合格词汇
        word_counts_root = os.path.join(args.out_root + '/word_counts.txt')
        with open(word_counts_root, 'w') as f:
            f.write('Total words: %d \n' % len(counts))
            f.write('Words in vocabulary: %d \n' % len(word_counts))
            f.write(str(word_counts))
        
        # 保存被过滤词汇
        word_counts_root = os.path.join(args.out_root + '/word_outs.txt')
        with open(word_counts_root, 'w') as f:
            f.write('Total words: %d \n' % len(counts))
            f.write('Words in vocabulary: %d \n' % len(words_out))
            f.write(str(words_out))

    # 创建有序词汇列表（仅单词，无频次）
    word_reverse = [w for (w,n) in word_counts]
    
    # 创建词汇到索引的映射字典
    vocab_dict = dict([(word, index) for (index, word) in enumerate(word_reverse)])
    
    # 创建Vocabulary对象（自定义类）
    vocab = Vocabulary(vocab_dict, len(vocab_dict))

    # Save word index as pickle form
    # 创建单词到索引的映射字典（同上，但单独实现）
    word_to_idx = {}
    for index, word in enumerate(word_reverse):
        word_to_idx[word] = index
    
    # 保存映射字典为pickle文件
    if write:
        with open(os.path.join(args.out_root, 'word_to_index.pkl'), 'wb') as f:
            pickle.dump(word_to_idx, f)

    # print('number of bad words: %d/%d = %.2f%%' % (bad_words, len(counts), bad_words * 100.0 / len(counts)))
    # print('number of words in vocab: %d/%d = %.2f%%' % (len(word_counts), len(counts), len(word_counts) * 100.0 / len(counts)))
    # print('number of Null: %d/%d = %.2f%%' % (bad_count, len(counts), bad_count * 100.0 / len(counts)))

    return vocab
#加载已保存的词汇表
def load_vocab(args):
     # 打开之前保存的词汇索引文件（二进制格式）
    with open(os.path.join(args.out_root, 'word_to_index.pkl'), 'rb') as f:
        # 从pickle文件中加载单词到索引的映射字典
        word_to_idx = pickle.load(f)
    # 使用加载的映射字典创建词汇表对象
    vocab = Vocabulary(word_to_idx, len(word_to_idx))
    return vocab


def process_metadata(split, data, args, write=True):
    """
    Wrap data into ImageMatadata form
    """
    # 初始化字典：映射图像ID到其所有描述
    id_to_captions = {}
    
    # 初始化列表：存储所有图像的元数据
    image_metadata = []
    
    # 计数器：总描述数量
    num_captions = 0
    
    # 计数器：处理的图像数量
    count = 0

    # 遍历数据集中的每个图像
    for img in data:
        # 图像计数器增加
        count += 1
        
        # 获取图像相对路径（非绝对路径）
        filepath = img['file_path']
        
        # 获取图像ID（减1是为了0-based索引）
        id = img['id'] - 1
        
        # 获取已处理的token化描述
        captions = img['processed_tokens']

        # print(captions)

        # 初始化当前ID的描述列表（如果不存在）
        id_to_captions.setdefault(id, [])
        
        # 将当前描述添加到该ID对应的列表中
        id_to_captions[id].append(captions)
        
        # 验证当前图像的分组与函数参数一致
        assert split == img['split'], 'error: wrong split'
        
        # 创建图像元数据对象并添加到列表
        image_metadata.append(ImageMetaData(id, filepath, captions, split))
        
        # 累加描述计数
        num_captions += len(captions)


    # 如果需要写入统计信息
    if write:
        # 打开元数据信息文件（追加模式）
        with open(os.path.join(args.out_root, 'metadata_info.txt'), 'a') as f:
            # 写入统计信息
            f.write("Total %d captions %d images %d identities in %s" % 
                   (num_captions, count, len(id_to_captions), split))
            f.write('\n')  # 换行

    return image_metadata


def process_decodedata(data, vocab):
    """
    Decode ImageMetaData to ImageDecodeData
    Each item in imagedecodedata has 2 captions. (len(captions_id) = 2)
    """
    # 初始化列表，用于存储转换后的数据
    image_decodedata = []
    
    # 遍历输入的图像元数据列表
    for img in data:
        # 获取图像路径
        image_path = img.image_path
    
        # 初始化列表，用于存储当前图像的所有描述的数字ID序列
        cap_to_vec = []

        # 遍历当前图像的所有描述
        for cap in img.captions:
            # 将文本描述转换为数字ID序列
            cap_to_vec.append(
                # 使用列表推导式：遍历描述中的每个单词
                [vocab.word_to_id(word) for word in cap]
            )
        
        # 创建解码后的数据对象并添加到结果列表
        image_decodedata.append(
            ImageDecodeData(
                img.id,          # 图像ID（保持不变）
                image_path,       # 图像路径（保持不变）
                cap_to_vec,       # 数字化的描述ID序列
                img.split         # 数据集分组（保持不变）
            )
        )

    # 返回转换后的数据列表
    return image_decodedata


def process_dataset(split, decodedata, args, write=True):
    # Process dataset
    
    # Arrange by caption in a sorted form
    dataset, label_range = create_dataset_sort(split, decodedata)

    data = write_dataset(split, dataset, args, write=write)
    return data
    

def create_dataset_sort(split, data):
    # 初始化一个空列表，用于存储排序后的图像数据
    images_sort = []
    # 初始化一个空字典，用于存储每个标签（ID）对应的索引范围
    label_range = {}
    # 初始化一个空字典，用于按标签（ID）分组存储图像数据
    images = {}
    
    # 遍历输入的所有图像数据
    for img in data:
        # 获取当前图像的ID作为标签
        label = img.id
        # 列表推导式：为每个caption_id创建一个新的ImageDecodeData对象
        # 这里将每个caption_id单独放入一个列表中，实现一对一的图像-描述对
        image = [ImageDecodeData(img.id, img.image_path, [caption_id], img.split) for caption_id in img.captions_id]
        
        # 如果该标签已存在于images字典中
        if label in images:
            # 将新创建的image对象列表添加到已有列表末尾
            images[label].extend(image)
            # 记录新添加的图像数量到label_range中
            label_range[label].append(len(image))
        # 如果该标签是第一次出现
        else:
            # 创建该标签的图像列表
            images[label] = image
            # 初始化该标签的范围列表，记录图像数量
            label_range[label] = [len(image)]

    # 初始化索引计数器为-1（因为后面会先累加再使用）
    index = -1
    # 遍历所有标签（ID）
    for label in images.keys():
        # 将当前标签的所有图像添加到排序列表中
        # 这确保了同一ID的所有图像在排序列表中连续排列
        images_sort.extend(images[label])
        
        # 记录当前索引作为该标签范围的起始点
        start = index
        
        # 遍历当前标签的所有图像组（每组对应一个caption_id）
        for index_image in range(len(label_range[label])):
            # 更新累积索引：将当前组的大小加到索引上
            # 这样label_range中存储的就是实际的结束索引位置
            label_range[label][index_image] += index
            # 更新当前索引为最新的结束位置
            index = label_range[label][index_image]
            
        # 将起始索引添加到label_range列表末尾
        # 这样label_range[label]的最后一个元素是起始索引，其他元素是结束索引
        label_range[label].append(start)

    # 返回排序后的图像列表和标签范围字典
    return images_sort, label_range


def write_dataset(split, data, args, label_range=None, write=True):
    # 函数定义，接受5个参数：
    # split: 数据集分割信息（如'train'、'val'、'test'）
    # data: 输入的数据集，包含图像对象列表
    # args: 命令行参数，包含输出路径等信息
    # label_range: 标签范围信息，默认为None
    # write: 是否将数据写入文件，默认为True
    
    """
    Separate each component
    Write dataset into binary file
    """
    # 函数文档字符串，说明函数功能：分离各个组件并将数据集写入二进制文件
    
    caption_id = []
    # 初始化空列表，用于存储所有图像的caption ID
    
    images_path = []
    # 初始化空列表，用于存储所有图像的路径
    
    labels = []
    # 初始化空列表，用于存储所有图像的标签（ID）

    for img in data:
        # 遍历输入数据中的每个图像对象
        
        assert len(img.captions_id) == 1
        # 断言确保每个图像对象只有一个caption ID，这是前面处理的结果
        
        caption_id.append(img.captions_id[0])
        # 将当前图像的caption ID添加到caption_id列表中
        
        labels.append(img.id)
        # 将当前图像的ID添加到labels列表中
        
        images_path.append(img.image_path)
        # 将当前图像的路径添加到images_path列表中

    #N = len(images)
    # 注释掉的代码，原本可能用于计算图像数量
    
    data = {'caption_id': caption_id, 'labels':labels, 'images_path':images_path}
    # 创建一个字典，包含三个键值对：caption_id、labels和images_path，分别对应前面收集的三个列表
    
    if write:
        # 如果write参数为True，则将数据写入文件
        
        if label_range is not None:
            # 如果提供了label_range参数（不为None）
            
            data['label_range'] = label_range
            # 将label_range添加到data字典中
            
            pickle_root = os.path.join(args.out_root, split + '_sort.pkl')
            # 构建排序后的pickle文件路径，格式为：args.out_root/split_sort.pkl
        else:
            # 如果没有提供label_range参数
            
            pickle_root = os.path.join(args.out_root, split + '.pkl')
            # 构建普通的pickle文件路径，格式为：args.out_root/split.pkl
            
        # Write caption_id and labels as pickle form
        # 注释：以pickle形式写入caption_id和labels
        
        with open(pickle_root, 'wb') as f:
            # 以二进制写入模式打开pickle文件
            
            pickle.dump(data, f)
            # 使用pickle.dump将data字典序列化并写入文件
            
        print('Save dataset') 
        # 打印提示信息，表示数据集已保存

    return data
    # 返回处理后的数据字典，包含caption_id、labels、images_path和可能的label_range
    


def generate_split(args):
    # 函数定义，接受一个参数 args（命令行参数对象）
    # 该函数的作用是根据数据集的划分信息生成训练集、验证集和测试集

    with open(args.json_root,'r') as f:
        # 打开 args.json_root 指定的 JSON 文件，以只读模式
        imgs = json.load(f)
        # 使用 json.load 加载 JSON 文件内容到 imgs 变量中
        # imgs 是一个包含所有图像信息的列表，每个元素是一个字典
    
    # print(imgs)
    # 注释掉的代码，原本用于打印 imgs 内容进行调试
    
    # process caption
    # 注释：处理图像说明文字（caption）
    
    if not args.load_tokens:
        # 如果 args.load_tokens 为 False（即不从 JSON 中加载已处理的 tokens）
        process_captions(imgs)
        # 调用 process_captions 函数处理 imgs 中的所有图像说明文字
        # 这个函数可能会为每个图像添加处理后的 tokens
    
    # else:
    #     print('Tokens in json preserved...')
    #     print('DEBUG', [im['processed_tokens'] for im in imgs[:10]])
    # 注释掉的代码，原本用于在 args.load_tokens 为 True 时打印调试信息
    # 表示保留 JSON 中的 tokens，并打印前 10 个图像的 processed_tokens
    
    val_data = []
    # 初始化空列表，用于存储验证集数据
    
    train_data = []
    # 初始化空列表，用于存储训练集数据
    
    test_data = []
    # 初始化空列表，用于存储测试集数据
    
    for img in imgs:
        # 遍历 imgs 中的每个图像对象
        
        if img['split'] == 'train':
            # 如果图像的 split 属性为 'train'（表示该图像属于训练集）
            train_data.append(img)
            # 将该图像添加到 train_data 列表中
            
        elif img['split'] =='val':
            # 如果图像的 split 属性为 'val'（表示该图像属于验证集）
            val_data.append(img)
            # 将该图像添加到 val_data 列表中
            
        else:
            # 如果图像的 split 属性既不是 'train' 也不是 'val'（默认为测试集）
            test_data.append(img)
            # 将该图像添加到 test_data 列表中
    
    write_json(train_data, os.path.join(args.out_root, 'train_reid.json'))
    # 调用 write_json 函数，将 train_data 写入到 args.out_root/train_reid.json 文件中
    
    write_json(val_data, os.path.join(args.out_root, 'val_reid.json'))
    # 调用 write_json 函数，将 val_data 写入到 args.out_root/val_reid.json 文件中
    
    write_json(test_data, os.path.join(args.out_root, 'test_reid.json'))
    # 调用 write_json 函数，将 test_data 写入到 args.out_root/test_reid.json 文件中

    return [train_data, val_data, test_data]
    # 返回一个包含训练集、验证集和测试集数据的列表

def load_split(args):
    
    data = []
    splits = ['train', 'val', 'test']
    for split in splits:
        split_root = os.path.join(args.out_root, split + '_reid.json')
        with open(split_root, 'r') as f:
            split_data = json.load(f)
        data.append(split_data)

    return data


def process_data(args):   
    if args.load_split:
        train_data, val_data, test_data = load_split(args)      
    else:
        train_data, val_data, test_data = generate_split(args)

    if args.load_vocab:   
        vocab = load_vocab(args)
    else:
        vocab = build_vocab(train_data, args)

    # Transform original data to Imagedata form.
    train_metadata = process_metadata('train', train_data, args)
    val_metadata = process_metadata('val', val_data, args)
    test_metadata = process_metadata('test', test_data, args)
     
    
    # Decode Imagedata to index caption and replace image file_root with image vecetor.
    train_decodedata = process_decodedata(train_metadata, vocab)
    val_decodedata = process_decodedata(val_metadata, vocab)
    test_decodedata = process_decodedata(test_metadata, vocab)
    
    
    process_dataset('train', train_decodedata, args)
    process_dataset('val', val_decodedata, args)
    process_dataset('test', test_decodedata, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Command for data preprocessing')
    parser.add_argument('--img_root', type=str, default='E:\deep code learning\Datasets\CUHK-PEDES/imgs')
    parser.add_argument('--json_root', type=str, default='E:\deep code learning\Datasets\CUHK-PEDES/reid_raw.json')
    parser.add_argument('--out_root', type=str, default='D:\Firefox download\SAF-main\cuhkpedes/processed_data')
    parser.add_argument('--min_word_count', type=int, default=0)
    parser.add_argument('--default_image_size', type=int, default=224)
    parser.add_argument('--load_split', action='store_true')
    parser.add_argument('--load_tokens', action='store_true')
    parser.add_argument('--load_vocab', action='store_true')
    parser.add_argument('--remove_stopwords', type=str, default=None)
    parser.add_argument('--keep_symbol', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    makedir(args.out_root)
    process_data(args)