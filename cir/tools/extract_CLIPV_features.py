import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor

# 引入项目配置
import sys
sys.path.append(os.getcwd()) 
from cir.modeling.demo_model import GaitCIRModel
import cir.config as cfg

# ================= ⚙️ 核心配置区域 =================
# 【开关】True = 去除背景(变黑); False = 保留原图背景
USE_MASK = True  

# 输入路径
DATA_ROOT = cfg.DATASET_ROOT
RGB_ROOT = os.path.join(DATA_ROOT, 'RGB')
MASK_ROOT = os.path.join(DATA_ROOT, 'Mask')
JSON_PATH = cfg.TRAIN_JSON 

# 输出路径：自动根据开关决定存哪里，避免混淆
if USE_MASK:
    OUTPUT_ROOT = '/root/autodl-tmp/CASIA-B-Processed/CLIP_feature_Masked'
    print("🎭 Mode: MASKED (Background removed)")
else:
    OUTPUT_ROOT = '/root/autodl-tmp/CASIA-B-Processed/CLIP_feature'
    print("🖼️ Mode: UNMASKED (Original background kept)")

# 其他参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==================================================

def load_and_preprocess_frames(seq_path, processor):
    full_seq_dir = os.path.join(RGB_ROOT, seq_path)
    if not os.path.isdir(full_seq_dir): return None

    frame_names = sorted([f for f in os.listdir(full_seq_dir) if f.endswith('.jpg')])
    if not frame_names: return None

    images = []
    for frame_name in frame_names:
        # 1. 读取 RGB
        rgb_path = os.path.join(full_seq_dir, frame_name)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # 2. 【关键逻辑】根据开关决定是否应用 Mask
        if USE_MASK:
            mask_name = frame_name.replace('.jpg', '.png')
            mask_path = os.path.join(MASK_ROOT, seq_path, mask_name)
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                mask_img = mask_img.astype(np.float32) / 255.0
                mask_img = mask_img[:, :, np.newaxis]
                # 融合：背景变黑
                rgb_img = (rgb_img * mask_img).astype(np.uint8)
        
        images.append(Image.fromarray(rgb_img))

    if not images: return None

    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values']

@torch.no_grad()
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 加载模型
    print(f"🚀 Loading Model: {cfg.MODEL_ID}")
    model = GaitCIRModel(cfg.MODEL_ID).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(cfg.MODEL_ID)

    # 读取 JSON 获取序列列表
    print(f"📂 Scanning JSON: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    unique_seqs = set()
    for item in data:
        unique_seqs.add(item['ref']['seq_path'])
        unique_seqs.add(item['tar']['seq_path'])
    
    sorted_seqs = sorted(list(unique_seqs))
    print(f"✅ Found {len(sorted_seqs)} unique sequences.")
    print(f"💾 Saving to: {OUTPUT_ROOT}")

    # 开始提取
    for seq_path in tqdm(sorted_seqs):
        save_path = os.path.join(OUTPUT_ROOT, seq_path + ".pt")
        save_dir = os.path.dirname(save_path)
        
        if os.path.exists(save_path): continue
            
        os.makedirs(save_dir, exist_ok=True)

        pixel_values = load_and_preprocess_frames(seq_path, processor)
        if pixel_values is None: continue
            
        pixel_values = pixel_values.to(DEVICE)
        feats = model.extract_img_feature(pixel_values)
        
        # 存盘
        torch.save(feats.cpu(), save_path)

    print("🎉 Done!")

if __name__ == '__main__':
    main()