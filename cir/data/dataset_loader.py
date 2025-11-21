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
                 json_path,              # 主索引文件路径
                 data_root,              # 图片数据根目录
                 split_config_path=None, # 分割配置文件路径
                 mode='train',           # 模式: 'train' 或 'test'
                 max_frames=1,           # 采样帧数 (训练1, 测试8)
                 transform=None, 
                 subject_token="the person", 
                 return_static=False):
        
        self.mode = mode
        self.max_frames = max_frames
        self.transform = transform
        self.subject_token = subject_token
        self.return_static = return_static
        self.rgb_root = os.path.join(data_root, 'RGB')
        self.mask_root = os.path.join(data_root, 'Mask')

        # === 1. 加载主数据 ===
        print(f"Loading Master JSON from {json_path} ...")
        with open(json_path, 'r') as f:
            all_data = json.load(f)
            
        # === 2. 应用分割过滤 (Split Filtering) ===
        if split_config_path and os.path.exists(split_config_path):
            print(f"Applying Split Config: {split_config_path} (Mode: {mode})")
            with open(split_config_path, 'r') as f:
                split_cfg = json.load(f)
            
            # 获取允许的 ID 列表 (转为 set 加速查找)
            if mode == 'train':
                allowed_ids = set(split_cfg['TRAIN_SET'])
            elif mode == 'test':
                allowed_ids = set(split_cfg['TEST_SET'])
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")
            
            # 核心过滤逻辑：保留 sid 在允许列表里的条目
            # 注意：确保 JSON 里的 sid 和 Config 里的格式一致 (都是字符串)
            self.data = [item for item in all_data if str(item['sid']) in allowed_ids]
            
            print(f"✅ Filter Applied: {len(all_data)} -> {len(self.data)} triplets kept.")
        else:
            print("⚠️ Warning: No split config provided. Using ALL data!")
            self.data = all_data

    def _load_frames(self, rel_seq_path):
        """加载帧逻辑 (保持不变)"""
        rgb_seq_dir = os.path.join(self.rgb_root, rel_seq_path)
        if not os.path.isdir(rgb_seq_dir): return []
        
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
                images.append(Image.new('RGB', (224, 224)))
                continue
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # 读取 Mask 并融合
            mask_name = frame_name.replace('.jpg', '.png')
            mask_path = os.path.join(self.mask_root, rel_seq_path, mask_name)
            
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # 二值化保险
                _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                mask_img = mask_img.astype(np.float32) / 255.0
                mask_img = mask_img[:, :, np.newaxis]
                rgb_img = (rgb_img * mask_img).astype(np.uint8)
            
            pil_img = Image.fromarray(rgb_img)
            if self.transform:
                pil_img = self.transform(pil_img)
            images.append(pil_img)
            
        return images

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            ref_imgs = self._load_frames(item['ref']['seq_path'])
            tar_imgs = self._load_frames(item['tar']['seq_path'])
            
            # 如果是训练单帧，直接解包
            if self.mode == 'train' and self.max_frames == 1:
                if ref_imgs: ref_out = ref_imgs[0]
                else: raise ValueError("Empty ref frames")
                
                if tar_imgs: tar_out = tar_imgs[0]
                else: raise ValueError("Empty tar frames")
            else:
                # 测试多帧，保持列表
                ref_out = ref_imgs
                tar_out = tar_imgs

            caption = item['caption'].replace("{subject}", self.subject_token)
            
            result = {
                "ref_imgs": ref_out,
                "tar_imgs": tar_out,
                "text": caption,
                "task": item['task'],
                # 传递元数据用于严格评测
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
            # print(f"Error loading index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.data)-1))

    def __len__(self):
        return len(self.data)