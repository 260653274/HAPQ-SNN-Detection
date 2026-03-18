import os
import py7zr
from pathlib import Path

# 定义路径
base_dir = Path("datasets")
gen1_dir = base_dir / "Gen1"
categories = ["train", "val", "test"]

# 创建目录
for cat in categories:
    (gen1_dir / cat).mkdir(parents=True, exist_ok=True)

# 解压函数
def extract_files(category):
    # 查找匹配的文件，例如 train_a.7z, train_b.7z
    files = sorted(list(base_dir.glob(f"{category}_*.7z")))
    target_dir = gen1_dir / category
    
    if not files:
        print(f"No files found for category: {category}")
        return

    print(f"Processing {category} files: {[f.name for f in files]}")
    
    for f in files:
        print(f"Extracting {f} to {target_dir}...")
        try:
            with py7zr.SevenZipFile(f, mode='r') as z:
                z.extractall(path=target_dir)
        except Exception as e:
            print(f"Error extracting {f}: {e}")

if __name__ == "__main__":
    # 执行解压
    for cat in categories:
        extract_files(cat)

    print("\nDataset setup complete.")
    print(f"Directory structure created at: {gen1_dir.absolute()}")
