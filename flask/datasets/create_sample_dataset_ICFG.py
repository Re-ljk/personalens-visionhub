import json
import os
import argparse

def create_sample_dataset(input_json_path, output_json_path, sample_size=100):
    """
    从原始ICFG-PEDES.json中提取前几个样本创建测试数据集
    
    Args:
        input_json_path: 原始ICFG-PEDES.json文件路径
        output_json_path: 输出的小样本JSON文件路径
        sample_size: 要提取的样本数量
    """
    print(f"正在读取原始数据集: {input_json_path}")
    
    # 读取原始JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"原始数据集包含 {len(original_data)} 个样本")
    
    # 提取前sample_size个样本
    sample_data = original_data[:sample_size]
    
    print(f"提取前 {len(sample_data)} 个样本")
    
    # 统计各个split的数量
    split_counts = {}
    for item in sample_data:
        split = item.get('split', 'unknown')
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print("样本分布:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} 个样本")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # 保存小样本数据集
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"小样本数据集已保存到: {output_json_path}")
    
    # 显示第一个样本的结构
    if sample_data:
        print("\n第一个样本的结构:")
        first_sample = sample_data[0]
        for key, value in first_sample.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: [{type(value[0]).__name__}] (长度: {len(value)})")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")

def parse_args():
    parser = argparse.ArgumentParser(description='创建ICFG-PEDES小样本测试数据集')
    parser.add_argument('--input_json', type=str, 
                       default='/mnt/disk2/postgraduates/2024/lishuangle/Python/I2T/Multi_Sub_ReID/data/ICFG-PEDES/ICFG-PEDES.json',
                       help='原始ICFG-PEDES.json文件路径')
    parser.add_argument('--output_json', type=str,
                       default='/mnt/disk2/postgraduates/2024/lishuangle/Python/I2T/Multi_Sub_ReID/data/ICFG-PEDES/ICFG-PEDES-sample.json',
                       help='输出的小样本JSON文件路径')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='要提取的样本数量 (默认: 100)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_json):
        print(f"错误: 输入文件不存在: {args.input_json}")
        exit(1)
    
    # 创建小样本数据集
    create_sample_dataset(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        sample_size=args.sample_size
    )
    
    print("\n完成! 现在你可以使用小样本数据集进行测试了。")
    print(f"要使用小样本数据集，请在运行预处理脚本时指定:")
    print(f"python preprocess_5SubSpace_ICFG.py --json_root {args.output_json}")