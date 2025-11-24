import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class GaitCIRDataset(Dataset):
    """
    通用数据加载器 (GaitCIR Universal Loader)
    功能：
      1. 支持 Image Mode (读生图 + 可选去背景)
      2. 支持 Feature Mode (读缓存特征，极速训练)
      3. 支持 OpenGait 风格的采样策略
    """
    def __init__(self, 
                 json_path,                  # 主索引 JSON 路径
                 data_root,                  # 原始图片根目录 (Image Mode 用)
                 split_config_path=None,     # 数据集划分配置
                 mode='train',               # 'train' 或 'test'
                 max_frames=4,               # 采样帧数
                 transform=None,             # 图像预处理 (Image Mode 用)
                 subject_token="the person", # 文本 Token 替换
                 return_static=False,        # 是否返回静态描述文本
                 use_features=False,         # 是否使用预提取特征
                 feature_root=None,          # 特征文件根目录
                 use_mask=True               # 是否去背景 (Image Mode 生效)
                 ):       
        
        self.mode = mode
        self.max_frames = max_frames
        self.transform = transform
        self.subject_token = subject_token
        self.return_static = return_static
        
        # === 模式配置 ===
        self.use_features = use_features
        self.feature_root = feature_root
        self.use_mask = use_mask
        
        # 路径检查
        if not self.use_features:
            self.rgb_root = os.path.join(data_root, 'RGB')
            self.mask_root = os.path.join(data_root, 'Mask')
        else:
            if self.feature_root is None:
                # 这里的路径检查可以防止空指针
                raise ValueError("❌ [Loader] Feature Root must be provided in Feature Mode!")

        # === 1. 加载索引数据 ===
        print(f"   Loading Index: {json_path}")
        with open(json_path, 'r') as f:
            all_data = json.load(f)
            
        # === 2. 数据划分过滤 (Split Filtering) ===
        if split_config_path and os.path.exists(split_config_path):
            with open(split_config_path, 'r') as f:
                split_cfg = json.load(f)
            
            # 根据模式选择对应的 ID 列表
            subset_key = 'TRAIN_SET' if mode == 'train' else 'TEST_SET'
            allowed_ids = set(split_cfg[subset_key])
            
            # 过滤数据
            self.data = [item for item in all_data if str(item['sid']) in allowed_ids]
            print(f"✅ Filter Applied: {len(all_data)} -> {len(self.data)} triplets kept.")
        else:
            self.data = all_data

    def _load_frames(self, rel_seq_path):
        """ [模式 A] 实时读取图片 (Image Mode) """
        rgb_seq_dir = os.path.join(self.rgb_root, rel_seq_path)
        if not os.path.isdir(rgb_seq_dir): return []
        
        all_frames = sorted([f for f in os.listdir(rgb_seq_dir) if f.endswith('.jpg')])
        if not all_frames: return []

        # 采样逻辑
        if self.mode == 'train':
            # 随机采样 (允许重复)
            selected_frames = random.choices(all_frames, k=self.max_frames)
        else:
            # 均匀采样
            indices = np.linspace(0, len(all_frames) - 1, self.max_frames, dtype=int)
            selected_frames = [all_frames[i] for i in indices]

        images = []
        for frame_name in selected_frames:
            rgb_path = os.path.join(rgb_seq_dir, frame_name)
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None: 
                # 坏图兜底: 返回黑图
                images.append(Image.new('RGB', (224, 224)))
                continue
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # Mask 去背景逻辑 (仅当 use_mask=True 且文件存在时)
            if self.use_mask:
                mask_name = frame_name.replace('.jpg', '.png')
                mask_path = os.path.join(self.mask_root, rel_seq_path, mask_name)
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                    mask_img = mask_img.astype(np.float32) / 255.0
                    mask_img = mask_img[:, :, np.newaxis]
                    rgb_img = (rgb_img * mask_img).astype(np.uint8)
            
            pil_img = Image.fromarray(rgb_img)
            if self.transform:
                pil_img = self.transform(pil_img)
            images.append(pil_img)
            
        return images

    def _load_features(self, rel_seq_path):
        """ [模式 B] 读取预提取特征 (Feature Mode) """
        # 路径拼凑: root/001/nm-01/090.pt
        feat_path = os.path.join(self.feature_root, rel_seq_path + ".pt")
        
        if not os.path.exists(feat_path): return None
        
        # map_location='cpu' 防止多进程 DataLoader 导致显存溢出
        features = torch.load(feat_path, map_location='cpu')
        total = features.size(0)
        
        if total == 0: return None

        # 采样逻辑
        if self.mode == 'train':
            # 训练：随机采样 N 帧
            if total >= self.max_frames:
                indices = sorted(random.sample(range(total), self.max_frames))
            else:
                indices = sorted(random.choices(range(total), k=self.max_frames))
            return features[indices] # [N, 512]
        else:
            # 测试：返回所有特征 (或者也可以做均匀采样)
            return features # [Total, 512]

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # === 1. 视觉数据加载 ===
            if self.use_features:
                ref_out = self._load_features(item['ref']['seq_path'])
                tar_out = self._load_features(item['tar']['seq_path'])
                if ref_out is None or tar_out is None:
                    raise ValueError(f"Missing features for {item['sid']}")
            else:
                ref_out = self._load_frames(item['ref']['seq_path'])
                tar_out = self._load_frames(item['tar']['seq_path'])

            # === 2. 文本指令处理 ===
            # 正向指令 (Ref -> Tar)
            caption = item['caption'].replace("{subject}", self.subject_token)
            
            # 逆向指令 (Tar -> Ref, 用于 Cycle Loss)
            raw_inv = item.get('caption_inv', "")
            caption_inv = raw_inv.replace("{subject}", self.subject_token) if raw_inv else ""
            
            result = {
                "ref_imgs": ref_out, # Tensor[T, 512] 或 List[PIL]
                "tar_imgs": tar_out,
                "text": caption,
                "text_inv": caption_inv,
                "task": item['task'],
                
                # 元数据 (用于测试评估)
                "sid": str(item['sid']),
                "cond": str(item['tar']['condition']),
                "view": str(item['tar']['view'])
            }
            
            # 静态描述 (可选)
            if self.return_static:
                result["ref_text"] = item['ref'].get('static_caption', "").replace("{subject}", self.subject_token)
                result["tar_text"] = item['tar'].get('static_caption', "").replace("{subject}", self.subject_token)
                
            return result
            
        except Exception:
            # 训练时如果遇到坏数据，随机重试另一个，保证 Robustness
            return self.__getitem__(random.randint(0, len(self.data)-1))

    def __len__(self):
        return len(self.data)