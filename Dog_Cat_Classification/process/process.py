import os
import shutil

# 源文件夹路径
source_folder = './data/val'

# 目标子文件夹路径
cats_folder = os.path.join(source_folder, 'cats')
dogs_folder = os.path.join(source_folder, 'dogs')

# 如果目标子文件夹不存在，则创建它们
os.makedirs(cats_folder, exist_ok=True)
os.makedirs(dogs_folder, exist_ok=True)

# 遍历源文件夹中的所有文件和目录
for item in os.listdir(source_folder):
    # 构造完整的文件或目录路径
    item_path = os.path.join(source_folder, item)
    
    # 跳过目录，只处理文件
    if os.path.isdir(item_path):
        continue
    
    # 根据文件名判断并移动文件
    if item.startswith('cat'):
        destination_path = os.path.join(cats_folder, item)
        shutil.move(item_path, destination_path)
    elif item.startswith('dog'):
        destination_path = os.path.join(dogs_folder, item)
        shutil.move(item_path, destination_path)

print("图片分类完成。")
