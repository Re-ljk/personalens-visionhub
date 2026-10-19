from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import subprocess
import base64
import shutil
from PIL import Image
import io
import glob

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"当前工作目录已设置为: {os.getcwd()}")

app = Flask(__name__)
# 启用CORS，允许所有域名访问
CORS(app)

# 或者更精确的配置，只允许特定域名和方法
# CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'], 
#      methods=['GET', 'POST'], 
#      allow_headers=['Content-Type', 'Authorization'])

@app.route('/api/v1/images_text_retrieval', methods=['POST'])
def images_text_retrieval():
    try:
        # 获取请求参数
        data = request.get_json()
        search_text = data.get('search')
        model_type = data.get('model')
        
        if not search_text or not model_type:
            return jsonify({'error': '缺少必要参数'}), 400
        
        if model_type in ('Ours', 'ours', 'AMDS', 'amds'):
            return handle_amds_model(search_text)
        elif model_type in ('Baseline', 'baseline', 'CFine', 'cfine'):
            return handle_cfine_model(search_text)
        else:
            return jsonify({'error': '不支持的模型类型'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handle_amds_model(search_text):
    """处理AMDS模型的请求"""
    try:
        target_dir = script_dir  # 直接指定正确的目录
        current_dir = os.getcwd()
        print(f"AMDS模型处理前：当前工作目录 = {current_dir}")
        print(f"AMDS模型处理前：目标工作目录 = {target_dir}")
        
        # 强制切换到正确目录
        os.chdir(target_dir)
        print(f"AMDS模型处理：已强制切换工作目录到 {os.getcwd()}")
        
        # 验证文件是否存在
        samples_json_path = './cuhkpedes/preprocessed_samples/test_reid.json'
        abs_samples_path = os.path.abspath(samples_json_path)
        print(f"AMDS模型：检查文件路径 {samples_json_path} -> {abs_samples_path}")
        print(f"AMDS模型：文件是否存在 = {os.path.exists(samples_json_path)}")
        
        # 第一步：从caption_all.json 中查找匹配的描述
        raw_json_path = './data/CUHK-PEDES/caption_all.json'
        with open(raw_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 查找匹配的条目并获取 file_path
        matched_file_paths = []
        for item in raw_data:
            if any(search_text.lower() in caption.lower() for caption in item['captions']):
                matched_file_paths.append(item['file_path'])
        
        if not matched_file_paths:
            return jsonify({'error': '在 reid_raw.json 中未找到匹配的描述'}), 404
        
        # 第二步：在 test_reid.json 中查找对应的 file_path
        # samples_json_path = './cuhkpedes/preprocessed_samples/test_reid_5_samples.json'
        samples_json_path = './cuhkpedes/preprocessed_samples/test_reid.json'
        with open(samples_json_path, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
        
        # 查找匹配的条目
        matched_items = []
        for item in samples_data:
            if item['file_path'] in matched_file_paths:
                matched_items.append(item)
        
        if not matched_items:
            return jsonify({'error': '在 test_reid.json 中未找到对应的 file_path'}), 404
        
        # 创建新的JSON文件
        new_json_path = './temp_query.json'
        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(matched_items, f, ensure_ascii=False, indent=2)
        
        # 修改test.py中的sample_json_path
        modify_test_py_for_amds(new_json_path)
        
        # -- 新方案：重定向输出到文件 --
        stdout_log_path = './temp_stdout.log'
        stderr_log_path = './temp_stderr.log'

        try:
            # 打开用于重定向的日志文件
            with open(stdout_log_path, 'wb') as stdout_log, open(stderr_log_path, 'wb') as stderr_log:
                print(f"执行test.py，查询文件: {new_json_path}, stdout日志: {stdout_log_path}, stderr日志: {stderr_log_path}")
                result = subprocess.run([
                    'python', 'test.py',
                    # '--batch_size', '5',
                    # '--max_length', '10'
                ], 
                stdout=stdout_log,  # 重定向stdout到文件
                stderr=stderr_log,  # 重定向stderr到文件
                timeout=600000000,       # 保持超时设置
                cwd='.'
                )

            # 从日志文件中读取输出内容
            with open(stdout_log_path, 'r', encoding='utf-8', errors='replace') as f:
                stdout_str = f.read()
            with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as f:
                stderr_str = f.read()

            print("--- test.py STDOUT (from file) ---")
            print(stdout_str)
            print("--- test.py STDERR (from file) ---")
            print(stderr_str)
            print(f"--- test.py 返回码: {result.returncode} ---")

            if result.returncode != 0:
                return jsonify({
                    'error': f'测试执行失败。请查看服务器日志获取详细的 stdout 和 stderr。',
                    'stdout': stdout_str,
                    'stderr': stderr_str,
                    'returncode': result.returncode
                }), 500
            
            # 处理结果图片
            results_dir = './top10_results_sample'
            print(f"准备处理结果图片，结果目录: {results_dir}")
            print(f"结果目录是否存在: {os.path.exists(results_dir)}")
            
            query_imgs = process_result_images(results_dir, matched_items)
            print(f"成功处理结果图片，共 {len(query_imgs)} 张")
            
            # 清理临时文件
            if os.path.exists(new_json_path):
                os.remove(new_json_path)
                print(f"已清理临时文件: {new_json_path}")
            
            return jsonify({
                'query_imgs': query_imgs,
                'gallery_size': 15370,  # 完整测试集大小
                'debug_info': {
                    'matched_items_count': len(matched_items),
                    'results_dir_exists': os.path.exists(results_dir),
                    'model': 'Ours'
                }
            })
            
        except subprocess.TimeoutExpired:
            # 超时后，日志文件可能包含部分输出
            with open(stdout_log_path, 'r', encoding='utf-8', errors='replace') as f:
                stdout_str = f.read()
            with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as f:
                stderr_str = f.read()

            print("--- test.py 执行超时 ---")
            print("--- TIMEOUT STDOUT (from file) ---")
            print(stdout_str)
            print("--- TIMEOUT STDERR (from file) ---")
            print(stderr_str)

            return jsonify({
                'error': '执行 test.py 超时',
                'stdout': stdout_str,
                'stderr': stderr_str
            }), 504

        except Exception as e:
            print(f"执行 test.py 时发生未知错误: {e}")
            return jsonify({'error': f'执行 test.py 时发生未知错误: {str(e)}'}), 500
        
        finally:
            # 确保临时日志文件被清理
            if os.path.exists(stdout_log_path):
                os.remove(stdout_log_path)
            if os.path.exists(stderr_log_path):
                os.remove(stderr_log_path)

    except subprocess.TimeoutExpired:
        print("test.py执行超时，正在终止进程...")
        return jsonify({"error": "test.py执行超时，请稍后重试"}), 500
    except Exception as e:
        print(f"处理AMDS模型时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

def handle_cfine_model(search_text):
    """处理CFine模型的请求"""
    try:
        # CFine模型的处理逻辑
        cfine_dir = './CFine'
        
        # 检查CFine目录是否存在
        if not os.path.exists(cfine_dir):
            print(f"CFine目录不存在: {cfine_dir}")
            return jsonify({'error': 'CFine目录不存在'}), 404
        
        print(f"CFine目录存在，准备切换到: {cfine_dir}")
        
        # 切换到CFine目录
        original_dir = os.getcwd()
        print(f"当前目录: {original_dir}")
        os.chdir(cfine_dir)
        print(f"已切换到CFine目录: {os.getcwd()}")
        
        try:
            # 读取CFine的JSON文件
            cfine_json_path = './data/CUHK-PEDES/caption_all.json'
            print(f"准备读取CFine数据文件: {cfine_json_path}")
            
            if not os.path.exists(cfine_json_path):
                print(f"CFine数据文件不存在: {cfine_json_path}")
                return jsonify({'error': f'CFine数据文件不存在: {cfine_json_path}'}), 404
                
            with open(cfine_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"成功读取CFine数据，共 {len(data)} 条记录")
            
            # 查找匹配的条目
            matched_items = []
            for item in data:
                if any(search_text.lower() in caption.lower() for caption in item['captions']):
                    matched_items.append(item)
            
            print(f"搜索文本: '{search_text}'")
            print(f"找到匹配的条目数量: {len(matched_items)}")
            
            if not matched_items:
                print("未找到匹配的描述")
                return jsonify({'error': '未找到匹配的描述'}), 404
            
            # 创建新的JSON文件
            new_json_path = './temp_query.json'
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(matched_items, f, ensure_ascii=False, indent=2)
            
            print(f"已创建临时查询文件: {new_json_path}")
            
            # -- 新方案：重定向输出到文件 --
            stdout_log_path = './temp_stdout.log'
            stderr_log_path = './temp_stderr.log'

            try:
                # 打开用于重定向的日志文件
                with open(stdout_log_path, 'wb') as stdout_log, open(stderr_log_path, 'wb') as stderr_log:
                    print(f"执行CFine test.py，查询文件: {new_json_path}, stdout日志: {stdout_log_path}, stderr日志: {stderr_log_path}")
                    result = subprocess.run([
                        'python', 'test.py'
                    ], 
                    stdout=stdout_log,  # 重定向stdout到文件
                    stderr=stderr_log,  # 重定向stderr到文件
                    timeout=600000000,       # 保持超时设置
                    cwd='.'
                    )

                # 从日志文件中读取输出内容
                with open(stdout_log_path, 'r', encoding='utf-8', errors='replace') as f:
                    stdout_str = f.read()
                with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as f:
                    stderr_str = f.read()

                print("--- CFine test.py STDOUT (from file) ---")
                print(stdout_str)
                print("--- CFine test.py STDERR (from file) ---")
                print(stderr_str)
                print(f"--- CFine test.py 返回码: {result.returncode} ---")

                if result.returncode != 0:
                    return jsonify({
                        'error': f'CFine测试执行失败。请查看服务器日志获取详细的 stdout 和 stderr。',
                        'stdout': stdout_str,
                        'stderr': stderr_str,
                        'returncode': result.returncode
                    }), 500

            except subprocess.TimeoutExpired:
                # 超时后，日志文件可能包含部分输出
                with open(stdout_log_path, 'r', encoding='utf-8', errors='replace') as f:
                    stdout_str = f.read()
                with open(stderr_log_path, 'r', encoding='utf-8', errors='replace') as f:
                    stderr_str = f.read()

                print("--- CFine test.py 执行超时 ---")
                print("--- TIMEOUT STDOUT (from file) ---")
                print(stdout_str)
                print("--- TIMEOUT STDERR (from file) ---")
                print(stderr_str)

                return jsonify({
                    'error': '执行 CFine test.py 超时',
                    'stdout': stdout_str,
                    'stderr': stderr_str
                }), 504

            except Exception as e:
                print(f"执行 CFine test.py 时发生未知错误: {e}")
                return jsonify({'error': f'执行 CFine test.py 时发生未知错误: {str(e)}'}), 500
            
            finally:
                # 确保临时日志文件被清理
                if os.path.exists(stdout_log_path):
                    os.remove(stdout_log_path)
                if os.path.exists(stderr_log_path):
                    os.remove(stderr_log_path)
            
            # 处理结果图片
            results_dir = './retrieval_results'
            print(f"准备处理结果图片，结果目录: {results_dir}")
            print(f"结果目录是否存在: {os.path.exists(results_dir)}")
            
            query_imgs = process_result_images(results_dir, matched_items)
            print(f"成功处理结果图片，共 {len(query_imgs)} 张")
            
            # 清理临时文件
            if os.path.exists(new_json_path):
                os.remove(new_json_path)
                print(f"已清理临时文件: {new_json_path}")
            
            return jsonify({
                'query_imgs': query_imgs,
                'gallery_size': 15370,  # 完整测试集大小
                'debug_info': {
                    'matched_items_count': len(matched_items),
                    'results_dir_exists': os.path.exists(results_dir),
                    'model': 'Baseline',
                    'cfine_dir': cfine_dir
                }
            })
            
        finally:
            # 恢复原始目录
            os.chdir(original_dir)
            print(f"已恢复到原始目录: {original_dir}")
        
    except subprocess.TimeoutExpired:
        print("CFine test.py执行超时，正在终止进程...")
        return jsonify({"error": "CFine test.py执行超时，请稍后重试"}), 500
    except Exception as e:
        print(f"处理CFine模型时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"CFine模型处理失败: {str(e)}"}), 500

def modify_test_py_for_amds(new_json_path):
    """修改test.py文件中的sample_json_path"""
    try:
        with open('test.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换sample_json_path - 使用正确的路径模式
        import re
        pattern = r'sample_json_path = "[^"]*"'
        replacement = f'sample_json_path = "{os.path.abspath(new_json_path)}"'
        new_content = re.sub(pattern, replacement, content)
        
        with open('test.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
            
    except Exception as e:
        print(f"修改test.py失败: {e}")

def resize_image_to_standard(image_path, target_width=151, target_height=201):
    """将图片调整为标准尺寸"""
    try:
        with Image.open(image_path) as img:
            # 调整图片尺寸为151x201
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # 将调整后的图片转换为字节数据
            img_buffer = io.BytesIO()
            # 保持原始格式，如果是JPEG就保存为JPEG，PNG就保存为PNG
            format = img.format if img.format else 'JPEG'
            resized_img.save(img_buffer, format=format)
            img_buffer.seek(0)
            
            return img_buffer.getvalue()
    except Exception as e:
        print(f"调整图片尺寸失败 {image_path}: {e}")
        # 如果调整失败，返回原始图片数据
        with open(image_path, 'rb') as f:
            return f.read()

def process_result_images(results_dir, matched_items):
    """处理结果图片，转换为data URI格式"""
    query_imgs = []
    
    # 首先添加查询图片
    query_img = process_query_image(matched_items)
    if query_img:
        query_imgs.append(query_img)
    
    if not os.path.exists(results_dir):
        return query_imgs
    
    # 获取所有结果图片
    image_files = glob.glob(os.path.join(results_dir, '**', '*.jpg'), recursive=True) + \
                  glob.glob(os.path.join(results_dir, '**', '*.png'), recursive=True)
    
    # 过滤掉非检索结果的图片，例如查询图片本身
    image_files = [f for f in image_files if 'rank' in os.path.basename(f)]
    
    # 临时存储检索结果图片信息
    retrieval_results = []

    for i, img_path in enumerate(image_files[:10]):  # 最多处理10张检索结果图片
        try:
            # 调整图片尺寸并读取数据
            img_data = resize_image_to_standard(img_path)
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 检测图像类型
            mime_type = get_image_mime_type(img_path)
            data_uri = f"data:{mime_type};base64,{img_base64}"
            
            # 从文件名或其他方式获取ID和分数
            filename = os.path.basename(img_path)
            img_id = extract_id_from_filename(filename)
            score = extract_score_from_filename(filename)
            
            retrieval_results.append({
                'id': img_id,
                'score': score,
                'pic': data_uri,
                'is_query': False  # 标识这是检索结果图片
            })
            
        except Exception as e:
            print(f"处理图片 {img_path} 失败: {e}")
            continue
    
    # 按照score降序排序（分数越高，相似度越高，排在前面）
    retrieval_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 将排序后的检索结果添加到query_imgs中
    query_imgs.extend(retrieval_results)
    
    return query_imgs

def get_image_mime_type(file_path):
    """根据文件扩展名获取MIME类型"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.jpg' or ext == '.jpeg':
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.gif':
        return 'image/gif'
    elif ext == '.bmp':
        return 'image/bmp'
    elif ext == '.webp':
        return 'image/webp'
    else:
        return 'image/jpeg'  # 默认为jpeg

def extract_score_from_filename(filename):
    """从文件名中提取分数"""
    try:
        # 文件名格式为: rank_X_label_Y_score_Z.jpg
        parts = filename.split('_')
        if 'score' in parts:
            score_idx = parts.index('score')
            if score_idx + 1 < len(parts):
                score_part = parts[score_idx + 1]
                # 移除文件扩展名 (.jpg)
                if '.' in score_part:
                    score_str = score_part.rsplit('.', 1)[0]  # 使用 rsplit 从右边分割，只分割一次
                else:
                    score_str = score_part
                return float(score_str)
        return 0.0
    except Exception as e:
        print(f"提取分数失败: {filename}, 错误: {e}")
        return 0.0

def extract_id_from_filename(filename):
    """从文件名中提取ID（label）"""
    try:
        # 文件名格式为: rank_X_label_Y_score_Z.jpg
        parts = filename.split('_')
        if 'label' in parts:
            label_idx = parts.index('label')
            if label_idx + 1 < len(parts):
                # 处理可能包含小数点的label值
                label_str = parts[label_idx + 1]
                return int(float(label_str))  # 先转换为float再转换为int
        return 0
    except Exception as e:
        print(f"提取ID失败: {filename}, 错误: {e}")
        return 0

def process_query_image(matched_items):
    """处理查询图片，转换为data URI格式"""
    if not matched_items:
        return None
    
    # 获取第一个匹配项的文件路径
    query_file_path = matched_items[0]['file_path']
    full_path = os.path.join('./data/CUHK-PEDES/imgs', query_file_path)
    
    try:
        if os.path.exists(full_path):
            # 调整图片尺寸并读取数据
            img_data = resize_image_to_standard(full_path)
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 检测图像类型并生成data URI
            mime_type = get_image_mime_type(full_path)
            data_uri = f"data:{mime_type};base64,{img_base64}"
            
            return {
                'id': matched_items[0].get('id', 0),  # 使用matched_items中的id
                'score': 1.0,  # 查询图片的相似度设为1.0
                'pic': data_uri,
                'is_query': True  # 标识这是查询图片
            }
    except Exception as e:
        print(f"处理查询图片 {full_path} 失败: {e}")
    
    return None

def normalize_similarity_scores(scores):
    """
    将余弦相似度归一化到[0,1]区间
    """
    # 方法1: 线性映射 [-1,1] -> [0,1]
    normalized = [(score + 1) / 2 for score in scores]
    
    # 方法2: Min-Max归一化
    min_score = min(scores)
    max_score = max(scores)
    if max_score > min_score:
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    else:
        normalized = [0.5] * len(scores)
    
    return normalized

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4523, debug=True, use_reloader=False)