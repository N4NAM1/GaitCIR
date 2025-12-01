import os
import cv2
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def process_and_save_view_rgb_only(rgb_view_dir, save_path, img_size=(224, 224)):
    """
    只读取 RGB 图片，不进行 Mask 去背景，直接打包为 PKL
    """
    if not os.path.exists(rgb_view_dir): return

    # 1. 过滤并排序图片 (支持 jpg, png, jpeg)
    frames = sorted([f for f in os.listdir(rgb_view_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not frames: return

    seq_images = []

    for frame_name in frames:
        rgb_path = os.path.join(rgb_view_dir, frame_name)
        
        # 读取图片
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: continue
        
        # BGR -> RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # 尺寸缩放 (可选，建议做，能大幅减小 PKL 体积)
        if img_size is not None:
            rgb_img = cv2.resize(rgb_img, img_size, interpolation=cv2.INTER_CUBIC)
            
        # 转为 PIL
        pil_img = Image.fromarray(rgb_img)
        seq_images.append(pil_img)

    # 2. 保存
    if seq_images:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(seq_images, f)

def main(args):
    rgb_root = args.data_root # 直接指向包含图片的根目录
    out_root = args.output_root
    
    print(f"🚀 开始转换 (RGB Only Mode)...")
    
    # 假设结构: Root/001/bg-01/000/xxx.jpg (CASIA-B 结构)
    # 如果你的结构不同，请微调这里的遍历逻辑
    if not os.path.exists(rgb_root):
        print(f"❌ 错误: 找不到目录 {rgb_root}")
        return

    subjects = sorted(os.listdir(rgb_root))
    
    for subject in tqdm(subjects, desc="Processing"):
        subj_path = os.path.join(rgb_root, subject)
        if not os.path.isdir(subj_path): continue
        
        conditions = sorted(os.listdir(subj_path))
        for cond in conditions:
            cond_path = os.path.join(subj_path, cond)
            if not os.path.isdir(cond_path): continue
            
            views = sorted(os.listdir(cond_path))
            for view in views:
                # 源目录: .../001/bg-01/000
                rgb_view_dir = os.path.join(subj_path, cond, view)
                
                # 目标文件: .../001/bg-01/000.pkl
                rel_path = Path(subject) / cond / view
                save_path = Path(out_root) / rel_path.with_suffix('.pkl')
                
                # 跳过已存在
                if os.path.exists(save_path) and not args.force:
                    continue
                
                process_and_save_view_rgb_only(rgb_view_dir, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/root/autodl-tmp/CASIA-B-Processed/RGB', type=str, help='原始数据集根目录')
    parser.add_argument('--output_root', default='/root/autodl-tmp/CASIA-B-Processed-pkl', type=str, help='输出 PKL 文件的根目录')
    parser.add_argument('--force', action='store_true', help='是否覆盖已存在的文件')
    args = parser.parse_args()
    
    main(args)

    # 示例运行命令：
    # python preprocess_pkl.py --data_root /path/to/CASIA-B-Processed --output_root /path/to/CASIA-B-PKLs