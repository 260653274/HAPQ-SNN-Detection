
import os
import numpy as np
from yolox.utils.psee_loader.io.psee_loader import PSEELoader
from tqdm import tqdm
import torch

def check_gen1_labels(data_dir):
    # 模拟 extract_labels 的部分逻辑来查找文件
    print(f"Checking data in {data_dir}")
    
    files = []
    # 递归查找所有 .npy 文件
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('_bbox.npy'):
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} bbox files.")
    
    invalid_files = []
    
    for file_path in tqdm(files):
        try:
            reader = PSEELoader(file_path)
            # 读取所有事件/bbox
            while not reader.done:
                # 这里的逻辑是简化版，直接读全部可能会内存爆炸，但 bbox 文件通常不大
                # 不过 PSEELoader 是流式读取的
                # 我们只检查 bbox 的值
                
                # 假设是 bbox 类型
                events = reader.load_n_events(10000)
                if len(events) == 0:
                    break
                
                # 检查 class_id
                if 'class_id' in events.dtype.names:
                    class_ids = events['class_id']
                    if np.any((class_ids < 0) | (class_ids >= 2)):
                        print(f"Invalid class_id in {file_path}: {np.unique(class_ids)}")
                        invalid_files.append(file_path)
                        break
                
                # 检查坐标
                if 'x' in events.dtype.names and 'y' in events.dtype.names and 'w' in events.dtype.names and 'h' in events.dtype.names:
                    x = events['x']
                    y = events['y']
                    w = events['w']
                    h = events['h']
                    
                    if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(w)) or np.any(np.isnan(h)):
                         print(f"NaN coordinates in {file_path}")
                         invalid_files.append(file_path)
                         break
                         
                    if np.any(np.isinf(x)) or np.any(np.isinf(y)) or np.any(np.isinf(w)) or np.any(np.isinf(h)):
                         print(f"Inf coordinates in {file_path}")
                         invalid_files.append(file_path)
                         break
                    
                    if np.any(w <= 0) or np.any(h <= 0):
                        # print(f"Non-positive width/height in {file_path}")
                        # 这可能是允许的（将被过滤），但值得注意
                        pass

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            invalid_files.append(file_path)

    print(f"Check complete. Found {len(invalid_files)} invalid files.")
    return invalid_files

if __name__ == "__main__":
    train_dir = "/home/xlang/EAS-SNN/datasets/Gen1/train"
    val_dir = "/home/xlang/EAS-SNN/datasets/Gen1/val"
    
    check_gen1_labels(train_dir)
    check_gen1_labels(val_dir)
