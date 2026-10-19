import os                                  # 操作系统接口
import sys                                 # 系统特定参数和函数
import time                                # 时间访问和转换
import shutil                              # 高级文件操作
import logging                             # 日志记录
import gc                                  # 垃圾回收
from set_gpus import setup_gpu
# 优先设置GPU，避免PyTorch提前初始化CUDA
setup_gpu()      
import torch                               # PyTorch深度学习框架
import torchvision.transforms as transforms # 图像转换工具
from utils.metric import AverageMeter, compute_topk_and_map  # 导入新的函数
from test_config import config             # 导入测试配置
from config import data_config, network_config, get_image_unique  # 导入配置函数
import numpy as np                         # 科学计算库
import math                                # 数学函数
import re                                  # 正则表达式
from matplotlib import pyplot as plt       # 绘图库
from datasets.pedes import CuhkPedes       # 导入行人数据集
from utils.visualize import visualize_image, visualize_img  # 导入可视化工具
from PIL import Image                      # 添加PIL导入
from sample_dataset import SampleDataset  # 导入新的样本数据集类
import torch.utils.data as data


def create_sample_data_loader(sample_json_path, image_root, batch_size=5, transform=None):
    """创建样本数据加载器"""
    from config import custom_collate_fn  
    
    dataset = SampleDataset(sample_json_path, image_root, transform)
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn  
    )
    return data_loader, dataset.unique


def create_full_test_data_loader(args):
    """创建完整测试数据集的数据加载器"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transforms = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    
    test_dataset = CuhkPedes(
        image_root=args.image_dir,
        anno_root=args.anno_dir,  
        split='test',
        max_length=args.max_length,
        transform=test_transforms,
        cap_transform=None  
    )
    
    from torch.utils.data import DataLoader
    from config import custom_collate_fn
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    return test_loader


def save_top10_retrieval_results_sample(query_loader, gallery_loader, network, args, unique_image, output_dir="./top10_results_sample", k=10):
    """
    为5个查询样本在完整gallery中保存top-10检索结果
    """
    # 创建输出目录前先清理旧文件
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    
    # 1. 处理查询样本（5个）
    query_feat_bank = torch.zeros(args.num_subspaces, 5, args.subspace_dim)
    query_labels_bank = torch.zeros(5)
    query_paths_bank = []
    query_captions_bank = []
    
    query_index = 0
    with torch.no_grad():
        for images, captions, labels in query_loader:
            # 添加调试信息
            print(f"Debug: captions type: {type(captions)}")
            print(f"Debug: captions length: {len(captions)}")
            if len(captions) > 0:
                print(f"Debug: captions[0] type: {type(captions[0])}")
                print(f"Debug: captions[0]: {captions[0]}")
                if isinstance(captions[0], list):
                    print(f"Debug: captions[0] length: {len(captions[0])}")
            
            # 预处理文本数据
            (
                tokens_all,
                segments_all,
                input_masks_all,
                caption_length_all,
            ) = network.module.language_model.pre_process_multiple(captions)

            # 将数据移至GPU
            tokens_all = tokens_all.cuda()
            segments_all = segments_all.cuda()
            input_masks_all = input_masks_all.cuda()
            images = images.cuda()
            labels = labels.cuda()
            interval = images.shape[0]

            # 边界检查
            if query_index + interval > 5:
                interval = 5 - query_index
                if interval <= 0:
                    break

            # 前向传播获取特征
            img_output, text_output, subspace_features = network(
                images, tokens_all, segments_all, input_masks_all
            )

            # 存储查询特征、标签和路径信息
            for i in range(len(img_output)):
                query_feat_bank[i][query_index : query_index + interval] = img_output[i][:interval]
            query_labels_bank[query_index : query_index + interval] = labels[:interval]
            
            # 获取当前批次的图片路径和文本描述
            for i in range(interval):
                dataset_idx = query_index + i
                if dataset_idx < len(query_loader.dataset):
                    img_path = query_loader.dataset.images[dataset_idx]
                    caption = query_loader.dataset.captions[dataset_idx]
                    
                    query_paths_bank.append(img_path)
                    query_captions_bank.append(caption)
            query_index += interval
            
            if query_index >= 5:
                break
    
    # 2. 处理完整的gallery
    max_gallery_size = 15370  # 完整测试集大小
    gallery_feat_bank = torch.zeros(args.num_subspaces, max_gallery_size, args.subspace_dim)
    gallery_labels_bank = torch.zeros(max_gallery_size)
    gallery_paths_bank = []
    gallery_captions_bank = []
    
    gallery_index = 0
    with torch.no_grad():
        for images, captions, labels in gallery_loader:
            # 预处理文本数据
            (
                tokens_all,
                segments_all,
                input_masks_all,
                caption_length_all,
            ) = network.module.language_model.pre_process_multiple(captions)

            # 将数据移至GPU
            tokens_all = tokens_all.cuda()
            segments_all = segments_all.cuda()
            input_masks_all = input_masks_all.cuda()
            images = images.cuda()
            labels = labels.cuda()
            interval = images.shape[0]

            # 边界检查
            if gallery_index + interval > max_gallery_size:
                interval = max_gallery_size - gallery_index
                if interval <= 0:
                    break

            # 前向传播获取特征
            img_output, text_output, subspace_features = network(
                images, tokens_all, segments_all, input_masks_all
            )

            # 存储gallery特征、标签和路径信息
            for i in range(len(text_output)):
                gallery_feat_bank[i][gallery_index : gallery_index + interval] = text_output[i][:interval]
            gallery_labels_bank[gallery_index : gallery_index + interval] = labels[:interval]
            
            # 获取当前批次的图片路径和文本描述
            for i in range(interval):
                dataset_idx = gallery_index + i
                if dataset_idx < len(gallery_loader.dataset):
                    # 根据数据集的split属性获取正确的图像路径和描述
                    if gallery_loader.dataset.split == 'test':
                        img_path = gallery_loader.dataset.test_images[dataset_idx]
                        caption = gallery_loader.dataset.test_captions[dataset_idx]
                    elif gallery_loader.dataset.split == 'val':
                        img_path = gallery_loader.dataset.val_images[dataset_idx]
                        caption = gallery_loader.dataset.val_captions[dataset_idx]
                    elif gallery_loader.dataset.split == 'train':
                        img_path = gallery_loader.dataset.train_images[dataset_idx]
                        caption = gallery_loader.dataset.train_captions[dataset_idx]
                    else:
                        raise ValueError(f"未知的数据集划分: {gallery_loader.dataset.split}")
                    
                    gallery_paths_bank.append(img_path)
                    gallery_captions_bank.append(caption)
            
            gallery_index += interval
            
            if gallery_index >= max_gallery_size:
                break
    
    # 3. 计算查询与gallery的相似度并检索top-10
    actual_query_size = query_index
    actual_gallery_size = gallery_index
    
    # 获取唯一查询图像的索引
    unique_image_indices = [i for i, is_unique in enumerate(unique_image) if is_unique and i < actual_query_size]
    
    if len(unique_image_indices) > 0:
        query = query_feat_bank[:, unique_image_indices, :]
        target_query = query_labels_bank[unique_image_indices]
        unique_query_paths = [query_paths_bank[i] for i in unique_image_indices]
        unique_query_captions = [query_captions_bank[i] for i in unique_image_indices]
    else:
        # 如果没有唯一图像，使用前几个样本
        num_samples = min(len(unique_image), actual_query_size)
        query = query_feat_bank[:, :num_samples, :]
        target_query = query_labels_bank[:num_samples]
        unique_query_paths = query_paths_bank[:num_samples]
        unique_query_captions = query_captions_bank[:num_samples]
    
    gallery = gallery_feat_bank[:, :actual_gallery_size, :]
    target_gallery = gallery_labels_bank[:actual_gallery_size]
    
    # 调整特征维度并计算相似度
    query = query.permute(1, 0, 2)  # [num_queries, num_subspaces, subspace_dim]
    gallery = gallery.permute(1, 0, 2)  # [gallery_size, num_subspaces, subspace_dim]
    
    test_query = query.reshape(query.size(0), -1)  # [num_queries, num_subspaces * subspace_dim]
    test_gallery = gallery.reshape(gallery.size(0), -1)  # [gallery_size, num_subspaces * subspace_dim]
    
    query_norm = test_query / test_query.norm(dim=1, keepdim=True)
    gallery_norm = test_gallery / test_gallery.norm(dim=1, keepdim=True)
    score = torch.matmul(query_norm, gallery_norm.t())  # [num_queries, gallery_size]
    
    # 4. 为每个查询图像保存top-10检索结果
    print(f"开始保存top-10检索结果到 {output_dir}")
    num_queries = score.size(0)
    
    for query_idx in range(num_queries):
        # 获取当前查询的相似度分数
        query_scores = score[query_idx]
        
        # 获取所有索引，按相似度排序
        _, sorted_indices = query_scores.sort(dim=0, descending=True)
        
        # 去重：选择不同的图像路径
        selected_indices = []
        seen_paths = set()
        
        for idx in sorted_indices:
            idx_item = idx.item()
            img_path = gallery_paths_bank[idx_item]
            
            # 如果这个图像路径还没有被选择过，就添加到结果中
            if img_path not in seen_paths:
                selected_indices.append(idx_item)
                seen_paths.add(img_path)
                
                # 如果已经选择了10张不同的图像，就停止
                if len(selected_indices) >= 10:
                    break
        
        # 创建查询特定的输出目录
        query_output_dir = os.path.join(output_dir, f"query_{query_idx}")
        # 如果查询目录已存在，先删除再创建
        if os.path.exists(query_output_dir):
            shutil.rmtree(query_output_dir)
        os.makedirs(query_output_dir)
        
        # 保存查询图像
        query_img_path = unique_query_paths[query_idx]
        query_full_path = os.path.join(args.image_dir, query_img_path)
        
        try:
            if os.path.exists(query_full_path):
                query_img = Image.open(query_full_path)
                query_save_path = os.path.join(query_output_dir, f"query_image.jpg")
                query_img.save(query_save_path)
                print(f"保存查询图像: {query_save_path}")
            else:
                print(f"查询图像不存在: {query_full_path}")
            
            # 保存查询图像的描述
            with open(os.path.join(query_output_dir, "query_caption.txt"), "w", encoding="utf-8") as f:
                f.write(f"Query Image: {query_img_path}\n")
                f.write(f"Query Label: {target_query[query_idx].item()}\n")
                f.write("Query Captions:\n")
                for i, cap in enumerate(unique_query_captions[query_idx]):
                    f.write(f"Caption {i+1}: {cap}\n")
        except Exception as e:
            print(f"保存查询图像时出错: {e}")
        
        # 保存top-k检索结果（使用去重后的索引）
        results_info = []
        for rank, gallery_idx in enumerate(selected_indices):
            # 获取检索到的图像路径和信息
            retrieved_img_path = gallery_paths_bank[gallery_idx]
            retrieved_full_path = os.path.join(args.image_dir, retrieved_img_path)
            retrieved_label = target_gallery[gallery_idx].item()
            similarity_score = query_scores[gallery_idx].item()
            
            # 保存检索到的图像
            try:
                if os.path.exists(retrieved_full_path):
                    retrieved_img = Image.open(retrieved_full_path)
                    # 修改文件名格式，包含Label ID
                    retrieved_save_path = os.path.join(query_output_dir, f"rank_{rank+1}_label_{int(retrieved_label)}_score_{similarity_score:.4f}.jpg")
                    retrieved_img.save(retrieved_save_path)
                    print(f"保存检索图像 rank {rank+1}: {retrieved_save_path}")
                else:
                    print(f"检索图像不存在: {retrieved_full_path}")
                
                # 记录结果信息 - 修复Unicode字符问题
                is_correct = "correct" if retrieved_label == target_query[query_idx].item() else "incorrect"
                results_info.append(f"Rank {rank+1}: {retrieved_img_path} (Label: {retrieved_label}, Score: {similarity_score:.4f}) {is_correct}")
                
                # 保存检索到的图像的描述
                if gallery_idx < len(gallery_captions_bank):
                    retrieved_captions = gallery_captions_bank[gallery_idx]
                    caption_file = os.path.join(query_output_dir, f"rank_{rank+1}_captions.txt")
                    with open(caption_file, "w", encoding="utf-8") as f:
                        f.write(f"Retrieved Image: {retrieved_img_path}\n")
                        f.write(f"Label: {retrieved_label}\n")
                        f.write(f"Similarity Score: {similarity_score:.4f}\n")
                        f.write("Captions:\n")
                        for i, cap in enumerate(retrieved_captions):
                            f.write(f"Caption {i+1}: {cap}\n")
                            
            except Exception as e:
                print(f"保存检索图像时出错: {e}")
                results_info.append(f"Rank {rank+1}: {retrieved_img_path} (Label: {retrieved_label}, Score: {similarity_score:.4f}) - 保存失败")
        
        # 保存检索结果信息
        try:
            with open(os.path.join(query_output_dir, "retrieval_results.txt"), "w", encoding="utf-8") as f:
                f.write(f"Top-{k} Retrieval Results for Query {query_idx}\n")
                f.write(f"Query Label: {target_query[query_idx].item()}\n")
                f.write("=" * 50 + "\n")
                for info in results_info:
                    f.write(info + "\n")
        except Exception as e:
            print(f"保存检索结果信息时出错: {e}")
        
        print(f"已保存查询 {query_idx} 的top-{k}检索结果")
    
    print(f"所有top-10检索结果已保存到 {output_dir}")


def test(data_loader, network, args, unique_image, epoch=0):
    """测试函数，用于评估模型性能
    
    Args:
        data_loader: 测试数据加载器
        network: 网络模型
        args: 参数配置
        unique_image: 唯一图像标记列表
        epoch: 当前训练轮次，默认为0
    """
    batch_time = AverageMeter()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    network.eval()
    max_size = 1565
    
    img_feat_bank = torch.zeros(args.num_subspaces, max_size, args.subspace_dim)
    text_feat_bank = torch.zeros(args.num_subspaces, max_size, args.subspace_dim)
    labels_bank = torch.zeros(max_size)
    index = 0
    
    with torch.no_grad():
        end = time.time()
        for images, captions, labels in data_loader:
            (
                tokens_all,
                segments_all,
                input_masks_all,
                caption_length_all,
            ) = network.module.language_model.pre_process_multiple(captions)

            tokens_all = tokens_all.cuda()
            segments_all = segments_all.cuda()
            input_masks_all = input_masks_all.cuda()
            images = images.cuda()
            labels = labels.cuda()
            interval = images.shape[0]

            if index + interval > max_size:
                print(f"Warning: index ({index}) + interval ({interval}) = {index + interval} > max_size ({max_size})")
                print(f"Truncating interval from {interval} to {max_size - index}")
                interval = max_size - index
                if interval <= 0:
                    print(f"Skipping batch: no space left (index={index}, max_size={max_size})")
                    break

            img_output, text_output, subspace_features = network(
                images, tokens_all, segments_all, input_masks_all
            )

            for i in range(len(img_output)):
                img_feat_bank[i][index : index + interval] = img_output[i][:interval]
                text_feat_bank[i][index : index + interval] = text_output[i][:interval]
            labels_bank[index : index + interval] = labels[:interval]
            
            batch_time.update(time.time() - end)
            end = time.time()
            index = index + interval
            
            if index >= max_size:
                print(f"Reached max_size ({max_size}), stopping data loading")
                break
        unique_image = torch.tensor(unique_image) == 1
        unique_image = unique_image[:index]
        
        result, score = compute_topk_and_map(
            img_feat_bank[:, unique_image, :],
            text_feat_bank,
            labels_bank[unique_image],
            labels_bank,
            [1, 5, 10],
            True,
        )
        
        (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            map_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
            map_t2i,
        ) = result
        
        return (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            map_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
            map_t2i,
            batch_time.avg,
        )


def main(args):
    """主函数，执行测试流程
    
    Args:
        args: 参数配置
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    # 使用5个样本数据作为查询
    sample_json_path = "./temp_query.json"
    
    # 创建查询数据加载器（5个样本）
    query_loader, unique_image = create_sample_data_loader(
        sample_json_path, 
        args.image_dir, 
        batch_size=1, 
        transform=test_transform
    )
    
    gallery_loader = create_full_test_data_loader(args)
    
    # 加载网络模型
    from models.model import Model
    network = Model(args)
    network = torch.nn.DataParallel(network).cuda()
    
    # 加载预训练模型
    checkpoint = torch.load(args.model_path)
    network.load_state_dict(checkpoint['network'])
    
    # 设置为评估模式
    network.eval()
    
    # 保存top-10检索结果
    output_dir = "./top10_results_sample"  # 定义输出目录
    save_top10_retrieval_results_sample(query_loader, gallery_loader, network, args, unique_image, output_dir)
    # print("5个样本的top-10检索结果保存完成！")
    
    # 清理显存和内存
    print("正在清理显存...")
    
    # 删除模型和数据加载器
    del network
    del query_loader
    del gallery_loader
    del checkpoint
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 强制垃圾回收
    gc.collect()
    
    print("显存清理完成！")


if __name__ == "__main__":
    args = config()
    main(args)
