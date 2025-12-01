import os
import json
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class GaitCIRDataset(Dataset):
    """
    Dual-PKL GaitCIR Loader (Final Robust Version)
    
    修复汇总:
      1. 路径适配: 支持 CASIA-B, CCPG, SUSTech1K
      2. 维度修复: 自动处理 Channel-First (3,H,W) -> (H,W,3)
      3. 采样修复: 支持 max_frames="all" 模式
    """
    def __init__(self, 
                 json_path, 
                 data_root, 
                 dataset_name,               # "CASIA-B", "CCPG", "SUSTech1K"
                 split_config_path=None, 
                 mode='train',
                 max_frames=4,
                 use_features=False, 
                 feature_root=None, 
                 use_mask=True,
                 transform=None,
                 subject_token="the person",
                 return_static=False
                 ):       
        
        self.mode = mode
        self.max_frames = max_frames
        self.transform = transform
        self.subject_token = subject_token
        self.return_static = return_static
        
        # === 1. 路径与模式配置 ===
        self.use_features = use_features
        self.feature_root = feature_root
        self.use_mask = use_mask
        self.dataset_name = dataset_name.upper()
        
        # 定义数据集的文件名策略
        if self.dataset_name == "CCPG":
            self.filename_replacement = {"rgb": "rgbs", "mask": "masks"}
        elif self.dataset_name == "SUSTECH1K":
            self.filename_replacement = "scan"
        else:
            self.filename_replacement = "append_pkl" 

        if not self.use_features:
            self.rgb_root = os.path.join(data_root, 'RGB')
            self.mask_root = os.path.join(data_root, 'Mask')

            if not os.path.exists(self.rgb_root):
                raise ValueError(f"❌ RGB root not found: {self.rgb_root}")
            
            if self.use_mask and not os.path.exists(self.mask_root):
                print(f"⚠️ Mask enabled but not found: {self.mask_root}. Disabling mask.")
                self.use_mask = False

        # === 2. 加载索引 ===
        print(f"   Loading Index: {json_path}")
        with open(json_path, 'r') as f:
            all_data = json.load(f)
            
        if split_config_path and os.path.exists(split_config_path):
            with open(split_config_path, 'r') as f:
                split_cfg = json.load(f)
            subset_key = 'TRAIN_SET' if mode == 'train' else 'TEST_SET'
            allowed_ids = set(split_cfg[subset_key])
            self.data = [item for item in all_data if str(item['sid']) in allowed_ids]
            print(f"✅ Filter Applied: {len(all_data)} -> {len(self.data)} triplets kept.")
        else:
            self.data = all_data

    def _get_pkl_path(self, root, rel_path, file_type="rgb"):
        """ 根据 Dataset 特性构建 PKL 文件路径 """
        
        # === A. SUSTech1K ===
        if self.dataset_name == "SUSTECH1K":
            dir_path = os.path.join(root, rel_path)
            if not os.path.isdir(dir_path): return None
            try:
                files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
                if not files: return None
                return os.path.join(dir_path, files[0])
            except Exception:
                return None

        # === B. CCPG ===
        elif self.dataset_name == "CCPG":
            if file_type == "rgb":
                return os.path.join(root, rel_path)
            elif file_type == "mask":
                if "rgbs" in rel_path:
                    new_rel_path = rel_path.replace("rgbs", "masks")
                    return os.path.join(root, new_rel_path)
                return os.path.join(root, rel_path.rsplit('.', 1)[0] + "_masks.pkl")

        # === C. CASIA-B ===
        else: 
            return os.path.join(root, rel_path + ".pkl")

    def _load_pkl(self, root, rel_path, file_type):
        """ 读取 PKL """
        path = self._get_pkl_path(root, rel_path, file_type)
        if path is None or not os.path.exists(path): 
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _load_dual_sequence(self, rel_seq_path):
        # 1. 读取 RGB
        rgb_list = self._load_pkl(self.rgb_root, rel_seq_path, file_type="rgb")
        if rgb_list is None or len(rgb_list) == 0: 
            return None
        
        # 2. 读取 Mask
        mask_list = []
        if self.use_mask:
            mask_list = self._load_pkl(self.mask_root, rel_seq_path, file_type="mask")
            if mask_list is not None and len(mask_list) > 0:
                min_len = min(len(rgb_list), len(mask_list))
                rgb_list = rgb_list[:min_len]
                mask_list = mask_list[:min_len]
            else:
                mask_list = [] 

        # 3. 采样 (🔥 核心修改：支持 "all")
        total = len(rgb_list)
        if total == 0: return None
        
        if self.mode == 'train':
            # 训练必须是 int
            frames_to_sample = self.max_frames if isinstance(self.max_frames, int) else 30
            indices = sorted([random.randint(0, total - 1) for _ in range(frames_to_sample)])
        else:
            # 测试支持 "all"
            if self.max_frames == "all" or self.max_frames is all: # 兼容字符串和内置函数(防呆)
                indices = np.arange(total)
            else:
                # 确保是 int
                frames_to_sample = int(self.max_frames)
                indices = np.linspace(0, total - 1, frames_to_sample, dtype=int)

        # 4. 融合与预处理
        final_imgs = []
        for idx in indices:
            pil_img = rgb_list[idx]
            
            # --- 维度与类型修正 ---
            if isinstance(pil_img, np.ndarray):
                # (3, H, W) -> (H, W, 3)
                if pil_img.ndim == 3 and pil_img.shape[0] == 3:
                    pil_img = pil_img.transpose(1, 2, 0)
                # (1, H, W) -> (H, W)
                elif pil_img.ndim == 3 and pil_img.shape[0] == 1:
                    pil_img = pil_img.squeeze(0)
                
                if pil_img.dtype != np.uint8:
                    pil_img = pil_img.astype(np.uint8)
                pil_img = Image.fromarray(pil_img)

            if self.use_mask and idx < len(mask_list):
                pil_mask = mask_list[idx]
                if isinstance(pil_mask, np.ndarray):
                    if pil_mask.ndim == 3: pil_mask = pil_mask.squeeze()
                    if pil_mask.dtype != np.uint8: pil_mask = pil_mask.astype(np.uint8)
                    pil_mask = Image.fromarray(pil_mask, mode='L')

                rgb_np = np.array(pil_img)
                mask_np = np.array(pil_mask)
                
                if mask_np.ndim == 2:
                    mask_np = mask_np[:, :, np.newaxis]
                mask_np = (mask_np > 127).astype(np.float32)
                
                # Resize Mask if needed
                if rgb_np.shape[:2] != mask_np.shape[:2]:
                    pil_mask = pil_mask.resize((rgb_np.shape[1], rgb_np.shape[0]), Image.NEAREST)
                    mask_np = np.array(pil_mask)
                    mask_np = (mask_np > 127).astype(np.float32)[:, :, np.newaxis]

                rgb_np = (rgb_np * mask_np).astype(np.uint8)
                pil_img = Image.fromarray(rgb_np)
            
            if self.transform:
                pil_img = self.transform(pil_img)
            
            final_imgs.append(pil_img)

        if len(final_imgs) > 0 and isinstance(final_imgs[0], torch.Tensor):
            return torch.stack(final_imgs)
            
        return final_imgs
    
    def _load_features(self, rel_seq_path):
        # Feature Mode
        path = os.path.join(self.feature_root, rel_seq_path + ".pt")
        if not os.path.exists(path): return None
        data = torch.load(path, map_location='cpu')
        total = data.size(0)
        if total == 0: return None
        
        if self.mode == 'train':
            frames_to_sample = self.max_frames if isinstance(self.max_frames, int) else 30
            indices = sorted([random.randint(0, total - 1) for _ in range(frames_to_sample)])
        else:
            # 🔥 支持 "all"
            if self.max_frames == "all" or self.max_frames is all:
                indices = np.arange(total)
            else:
                frames_to_sample = int(self.max_frames)
                indices = np.linspace(0, total - 1, frames_to_sample, dtype=int)
                
        return data[indices]

    def __getitem__(self, idx):
        retries = 0
        max_retries = 20
        
        while True:
            if retries > max_retries:
                raise RuntimeError(f"❌ Failed to load data after {max_retries} attempts.")
                
            item = self.data[idx]
            try:
                if self.use_features:
                    ref_out = self._load_features(item['ref']['seq_path'])
                    tar_out = self._load_features(item['tar']['seq_path'])
                else:
                    ref_out = self._load_dual_sequence(item['ref']['seq_path'])
                    tar_out = self._load_dual_sequence(item['tar']['seq_path'])

                if ref_out is None or tar_out is None:
                    raise ValueError(f"Missing data")

                caption = item['caption'].replace("{subject}", self.subject_token)
                raw_inv = item.get('caption_inv', "")
                caption_inv = raw_inv.replace("{subject}", self.subject_token) if raw_inv else ""
                
                result = {
                    "ref_imgs": ref_out, "tar_imgs": tar_out,
                    "text": caption, "text_inv": caption_inv,
                    "task": item['task'], "sid": str(item['sid']),
                    "cond": str(item['tar']['condition']), "view": str(item['tar']['view'])
                }
                if self.return_static:
                    result["ref_text"] = item['ref'].get('static_caption', "").replace("{subject}", self.subject_token)
                    result["tar_text"] = item['tar'].get('static_caption', "").replace("{subject}", self.subject_token)
                
                return result 
            
            except Exception as e:
                if self.mode == 'train':
                    idx = random.randint(0, len(self.data) - 1)
                    retries += 1
                else:
                    raise 

    def __len__(self):
        return len(self.data)