import os
import subprocess
import shutil


def auto_select_gpu(memory_threshold=500, util_threshold=5):
    """
    自动选择一个最空闲的 GPU。

    选择逻辑:
    1. 首先筛选出 "显存占用" 和 "GPU利用率" 均低于阈值的候选GPU。
    2. 在候选GPU中，优先选择 GPU利用率 最低的。
    3. 如果利用率相同，则选择 剩余显存 最多的。
    4. 如果没有满足阈值条件的GPU，则选择所有GPU中 剩余显存 最多的一个。

    :param memory_threshold: int, 显存占用阈值 (MB)。低于此值的GPU才被视为候选。
    :param util_threshold: int, GPU利用率阈值 (%)。低于此值的GPU才被视为候选。
    :return: str, 最优GPU的ID。如果无可用GPU或查询失败，返回'0'作为默认值。
    """
    # 检查 nvidia-smi 命令是否存在
    if not shutil.which('nvidia-smi'):
        print("警告: 未找到 'nvidia-smi' 命令，将使用默认GPU '0'。")
        return '0'

    try:
        # 查询GPU的ID、利用率、已用显存、总显存
        fields = ('index', 'utilization.gpu', 'memory.used', 'memory.total')
        cmd = ['nvidia-smi', f'--query-gpu={",".join(fields)}', '--format=csv,noheader,nounits']
        output = subprocess.check_output(cmd, universal_newlines=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: 执行 'nvidia-smi' 失败，将使用默认GPU '0'。")
        return '0'

    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        try:
            gpu_info = {
                'id': parts[0],
                'util': int(parts[1]),
                'mem_used': int(parts[2]),
                'mem_total': int(parts[3]),
                'mem_free': int(parts[3]) - int(parts[2]) # 计算剩余显存
            }
            gpus.append(gpu_info)
        except (ValueError, IndexError):
            print(f"警告: 解析行失败: '{line}'")
            continue

    if not gpus:
        print("警告: 未查询到任何有效的GPU信息，将使用默认GPU '0'。")
        return '0'

    # 筛选出满足条件的候选GPU
    candidate_gpus = [gpu for gpu in gpus if gpu['mem_used'] < memory_threshold and gpu['util'] < util_threshold]

    if candidate_gpus:
        # 优先按GPU利用率排序（越小越好），其次按剩余显存排序（越大越好）
        best_gpu = sorted(candidate_gpus, key=lambda g: (g['util'], -g['mem_free']))[0]
    else:
        # 如果没有完全满足条件的，就退而求其次，在所有GPU中选择剩余显存最多的
        print("提示: 没有找到完全空闲的GPU，将选择剩余显存最多的GPU。")
        best_gpu = sorted(gpus, key=lambda g: -g['mem_free'])[0]

    return best_gpu['id']

def setup_gpu():
    """
    调用GPU选择函数并设置环境变量。
    """
    gpu_id = auto_select_gpu()
    print(f">>> 自动选择最合适的 GPU: {gpu_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f">>> 已设置环境变量 CUDA_VISIBLE_DEVICES = {gpu_id}")
