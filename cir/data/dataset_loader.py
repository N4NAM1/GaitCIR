import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class GaitCIRDataset(Dataset):
    def __init__(self, 
                 json_path,                  # 主索引文件路径
                 data_root,                  # 图片数据根目录
                 split_config_path=None,     # 分割配置文件路径
                 mode='train',               # 模式: 'train' 或 'test'
                 max_frames=1,               # 采样帧数 (训练1, 测试8)
                 transform=None,             # 图像预处理函数
                 subject_token="the person", # 替换文本中的主体标记
                 return_static=False):       # 是否返回静态描述
        
        self.mode = mode
        self.max_frames = max_frames
        self.transform = transform
        self.subject_token = subject_token
        self.return_static = return_static
        
        # 【修改】只保留 RGB 路径，不再需要 Mask 路径
        self.rgb_root = os.path.join(data_root, 'RGB')
        # self.mask_root = os.path.join(data_root, 'Mask') # 已移除

        # === 1. 加载主数据 ===
        print(f"Dataset Mode: {mode} | Max Frames: {max_frames}")
        print(f"Loading Master JSON from {json_path} ...")
        with open(json_path, 'r') as f:
            all_data = json.load(f)
            
        # === 2. 应用分割过滤 (Split Filtering) ===
        if split_config_path and os.path.exists(split_config_path):
            print(f"Applying Split Config: {split_config_path}")
            with open(split_config_path, 'r') as f:
                split_cfg = json.load(f)
            
            # 获取允许的 ID 列表
            if mode == 'train':
                allowed_ids = set(split_cfg['TRAIN_SET'])
            elif mode == 'test':
                allowed_ids = set(split_cfg['TEST_SET'])
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")
            
            # 核心过滤逻辑
            self.data = [item for item in all_data if str(item['sid']) in allowed_ids]
            
            print(f"✅ Filter Applied: {len(all_data)} -> {len(self.data)} triplets kept.")
        else:
            print("⚠️ Warning: No split config provided. Using ALL data!")
            self.data = all_data

    def _load_frames(self, rel_seq_path):
        """加载帧逻辑：只加载纯净的 RGB 裁剪图"""
        rgb_seq_dir = os.path.join(self.rgb_root, rel_seq_path)
        if not os.path.isdir(rgb_seq_dir): return []
        
        # 获取该序列下所有 jpg 图片
        all_frames = sorted([f for f in os.listdir(rgb_seq_dir) if f.endswith('.jpg')])
        if not all_frames: return []

        # 根据模式决定采样策略
        if self.mode == 'train':
            # 训练：随机采样 (允许重复)
            selected_frames = random.choices(all_frames, k=self.max_frames)
        else:
            # 测试：均匀采样
            indices = np.linspace(0, len(all_frames) - 1, self.max_frames, dtype=int)
            selected_frames = [all_frames[i] for i in indices]

        images = []
        for frame_name in selected_frames:
            # 读取 RGB
            rgb_path = os.path.join(rgb_seq_dir, frame_name)
            rgb_img = cv2.imread(rgb_path)
            
            if rgb_img is None: 
                # 失败兜底：返回黑图防止 Crash
                images.append(Image.new('RGB', (224, 224)))
                continue
                
            # 转为 RGB (OpenCV 默认是 BGR)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # 【已移除】 Mask 读取与融合逻辑
            # mask_name = frame_name.replace('.jpg', '.png')
            # ... (fusion logic removed) ...
            
            pil_img = Image.fromarray(rgb_img)
            if self.transform:
                pil_img = self.transform(pil_img)
            images.append(pil_img)
            
        return images

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # 获取相对路径 (例如: 001/nm-01/090)
            # 确保 json 中的 path 是相对路径，不包含 root
            ref_imgs = self._load_frames(item['ref']['seq_path'])
            tar_imgs = self._load_frames(item['tar']['seq_path'])
            
            # 训练模式下，如果只采了一帧，直接解包 (T, C, H, W) -> (C, H, W)
            if self.mode == 'train' and self.max_frames == 1:
                if ref_imgs: ref_out = ref_imgs[0]
                else: raise ValueError(f"Empty ref frames for {item['ref']['seq_path']}")
                
                if tar_imgs: tar_out = tar_imgs[0]
                else: raise ValueError(f"Empty tar frames for {item['tar']['seq_path']}")
            else:
                ref_out = ref_imgs
                tar_out = tar_imgs

            # === 处理文本 ===
            caption = item['caption'].replace("{subject}", self.subject_token)
            
            raw_inv = item.get('caption_inv', "")
            if raw_inv:
                caption_inv = raw_inv.replace("{subject}", self.subject_token)
            else:
                caption_inv = ""
            
            result = {
                "ref_imgs": ref_out,
                "tar_imgs": tar_out,
                "text": caption,             
                "text_inv": caption_inv,     
                "task": item['task'],        
                
                "sid": str(item['sid']),
                "cond": str(item['tar']['condition']),
                "view": str(item['tar']['view'])
            }
            
            if self.return_static:
                ref_st = item['ref'].get('static_caption', "").replace("{subject}", self.subject_token)
                tar_st = item['tar'].get('static_caption', "").replace("{subject}", self.subject_token)
                result["ref_text"] = ref_st
                result["tar_text"] = tar_st
                
            return result
            
        except Exception as e:
            print(f"⚠️ Error loading index {idx} ({item.get('sid', 'unknown')}): {e}. Retrying...")
            return self.__getitem__(random.randint(0, len(self.data)-1))

    def __len__(self):
        return len(self.data)