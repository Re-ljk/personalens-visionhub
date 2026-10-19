import torch
from torch import nn
from .pytorch_transformers import BertModel, BertConfig, BertTokenizer


class Bert(nn.Module):
    """BERT文本特征提取器
    
    该类封装了BERT模型，用于从文本中提取特征表示。
    它提供了文本预处理和特征提取的功能。
    """
    def __init__(self,max_length = 100): 
        """初始化BERT模型
        
        加载预训练的BERT分词器、配置和模型权重
        """
        super(Bert, self).__init__()
        self.max_length = max_length

        # 初始化BERT分词器，用于将文本转换为token
        self.tokenizer = BertTokenizer.from_pretrained('./data/uncased_L-12_H-768_A-12/vocab.txt')
        # 加载BERT模型配置
        modelConfig = BertConfig.from_pretrained('./data/uncased_L-12_H-768_A-12/config.json')
        # 初始化BERT模型作为文本特征提取器
        self.textExtractor = BertModel.from_pretrained(
            './data/uncased_L-12_H-768_A-12/pytorch_model.bin', config=modelConfig)

    def pre_process2(self, texts):
        """处理单个文本
        
        将单个文本转换为token序列
        
        参数:
            texts: 单个文本字符串
            
        返回:
            tokens: 处理后的token列表
        """
        tokens, segments, input_masks, text_length = [], [], [], []
        text=texts
        # 添加特殊标记[CLS]和[SEP]
        text = '[CLS] ' + text + ' [SEP]'  
        # 将文本转换为token
        tokenized_text = self.tokenizer.tokenize(text)  # 维度: [seq_len]
        # 将token转换为ID
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)  # 维度: [seq_len]
        # 截断过长的序列
        if len(indexed_tokens) > 100:
            indexed_tokens = indexed_tokens[:100]
        # 填充到固定长度100
        for i in range(len(tokens),100):
            tokenized_text.append('[PAD]')
        tokens=tokenized_text
        return tokens
        
    def pre_process(self, texts):
        """批量处理文本
        
        将文本列表转换为模型输入所需的token、segment和mask张量
        
        参数:
            texts: 文本字符串列表
            
        返回:
            tokens: token ID张量，维度 [batch_size, 100]
            segments: segment ID张量，维度 [batch_size, 100]
            input_masks: 注意力掩码张量，维度 [batch_size, 100]
            text_length: 文本实际长度张量，维度 [batch_size]
        """
        tokens, segments, input_masks, text_length = [], [], [], []
        
        # 处理每个文本
        for text in texts:
            # print(text)
            # 添加特殊标记[CLS]和[SEP]
            text = '[CLS] ' + text + ' [SEP]'
            # 将文本转换为token
            tokenized_text = self.tokenizer.tokenize(text)  # 维度: [seq_len]
            # 将token转换为ID
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)  # 维度: [seq_len]
            # 截断过长的序列
            if len(indexed_tokens) > 100:
                indexed_tokens = indexed_tokens[:100]
                
            tokens.append(indexed_tokens)
            # 对于单个句子，segment ID全为0
            segments.append([0] * len(indexed_tokens))
            # 有效token的mask为1
            input_masks.append([1] * len(indexed_tokens))
            
        # 对每个序列进行填充，使其长度统一为100
        for j in range(len(tokens)):
            padding = [0] * (100 - len(tokens[j]))
            text_length.append(len(tokens[j])+3)  # 记录实际文本长度（加3可能是为了特殊标记）
            tokens[j] += padding  # 填充token序列
            segments[j] += padding  # 填充segment序列
            input_masks[j] += padding  # 填充mask序列
            
        # 转换为张量
        tokens = torch.tensor(tokens)  # 维度: [batch_size, 100]
        segments = torch.tensor(segments)  # 维度: [batch_size, 100]
        input_masks = torch.tensor(input_masks)  # 维度: [batch_size, 100]
        text_length = torch.tensor(text_length)  # 维度: [batch_size]

        return tokens, segments, input_masks, text_length

    def pre_process_multiple(self, texts_list):
        """批量处理多个文本列表（每个样本有5个句子）
        
        参数:
            texts_list: 文本列表的列表，形状为 [batch_size, 5]
            
        返回:
            all_tokens: 所有句子的token ID张量，维度 [batch_size, 5, 100]
            all_segments: 所有句子的segment ID张量，维度 [batch_size, 5, 100]
            all_input_masks: 所有句子的注意力掩码张量，维度 [batch_size, 5, 100]
            all_text_length: 所有句子的实际长度张量，维度 [batch_size, 5]
        """
        batch_size = len(texts_list)
        # 动态获取句子数量，而不是硬编码
        num_sentences = len(texts_list[0]) if batch_size > 0 else 1
        
        all_tokens = []
        all_segments = []
        all_input_masks = []
        all_text_length = []
        
        for batch_idx in range(batch_size):
            batch_tokens = []
            batch_segments = []
            batch_input_masks = []
            batch_text_length = []
            
            # 确保不会越界
            actual_sentences = len(texts_list[batch_idx])
            for sent_idx in range(min(num_sentences, actual_sentences)):
                text = texts_list[batch_idx][sent_idx]
                if not isinstance(text, str):
                    text = str(text)
                
                # 直接分词原始文本，不预先添加特殊标记
                tokens_a = self.tokenizer.tokenize(text)
                
                # 截断到合适长度，为[CLS]和[SEP]预留空间
                if len(tokens_a) > self.max_length - 2:
                    print(len(tokens_a))
                    tokens_a = tokens_a[0:(self.max_length - 2)]
                    
                
                # 构建完整的token序列
                tokens = []
                segment_ids = []
                
                # 添加[CLS]
                tokens.append("[CLS]")
                segment_ids.append(0)
                
                # 添加实际内容
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                
                # 添加[SEP]
                tokens.append("[SEP]")
                segment_ids.append(0)
                
                # 转换为ID
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                text_length = len(input_ids)
                
                # 填充到100长度
                while len(input_ids) < self.max_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                
                batch_tokens.append(input_ids)
                batch_segments.append(segment_ids)
                batch_input_masks.append(input_mask)
                batch_text_length.append(text_length)
            
            all_tokens.append(batch_tokens)
            all_segments.append(batch_segments)
            all_input_masks.append(batch_input_masks)
            all_text_length.append(batch_text_length)

        return (
            torch.tensor(all_tokens, dtype=torch.long),
            torch.tensor(all_segments, dtype=torch.long),
            torch.tensor(all_input_masks, dtype=torch.long),
            torch.tensor(all_text_length, dtype=torch.long)
        )

    def forward(self, tokens, segments, input_masks):
        """前向传播函数
        
        使用BERT模型提取文本特征
        
        参数:
            tokens: token ID张量，维度 [batch_size, 100]
            segments: segment ID张量，维度 [batch_size, 100]
            input_masks: 注意力掩码张量，维度 [batch_size, 100]
            
        返回:
            output[0]: BERT输出的序列表示，维度 [batch_size, 100, 768]
            att_scores: 注意力分数，维度 [batch_size, num_layers, num_heads, 100, 100]
        """
        # 调用BERT模型提取特征
        # output[0]是序列表示，维度为[batch_size, seq_len, hidden_size]
        # att_scores是注意力分数，维度为[batch_size, num_layers, num_heads, seq_len, seq_len]
        output, att_scores = self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)

        return output[0], att_scores


