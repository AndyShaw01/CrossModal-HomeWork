import numpy as np

def split_large_npy_file(file_path, num_parts):
    # 使用内存映射打开文件
    mmap = np.load(file_path, mmap_mode='r')
    total_elements = mmap.shape[0]
    part_size = total_elements // num_parts
    
    for i in range(num_parts):
        start = i * part_size
        # 确保最后一个部分包含所有剩余的元素
        if i == num_parts - 1:
            end = total_elements
        else:
            end = start + part_size
        
        # 读取部分数据
        part = mmap[start:end]
        
        # 保存部分到新的文件
        np.save(f'part_{i}.npy', part)

# 使用此函数进行分割
split_large_npy_file('./download/train_ims.npy', 10)
