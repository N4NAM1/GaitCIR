import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class GaitCIRDataset(Dataset):
    def __init__(self, json_path, data_root, transform=None, subject_token="the person", return_static=False):
        """
        Args:
            json_path (str): Step 04 生成的最终训练 JSON 路径
            data_root (str): 数据集根目录 (包含 'RGB' 和 'Mask' 子文件夹)
            transform (callable): CLIP 官方的预处理转换 (Resize, CenterCrop, Norm...)
            subject_token (str): 将 {subject} 替换为什么？
                                 - 普通训练用 "the person"
                                 - 伪词训练用 "$S_*$"
            return_static (bool): 是否同时返回静态描述 (用于辅助 Loss)
        """
        print(f"Loading dataset from {json_path} ...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.rgb_root = os.path.join(data_root, 'RGB')
        self.mask_root = os.path.join(data_root, 'silhouettes')
        self.transform = transform
        self.subject_token = subject_token
        self.return_static = return_static
        
        print(f"Dataset loaded. Total triplets: {len(self.data)}")

    def _load_random_frame(self, rel_seq_path):
        """
        从序列文件夹中随机读取一帧，并执行 Masked RGB 融合
        """
        # 1. 定位 RGB 序列文件夹
        rgb_seq_dir = os.path.join(self.rgb_root, rel_seq_path)
        
        # 鲁棒性检查
        if not os.path.isdir(rgb_seq_dir):
            raise FileNotFoundError(f"Sequence dir not found: {rgb_seq_dir}")
            
        frames = [f for f in os.listdir(rgb_seq_dir) if f.endswith('.jpg')]
        if not frames:
            raise FileNotFoundError(f"No frames in {rgb_seq_dir}")

        # 2. 随机采样一帧
        frame_name = random.choice(frames)
        
        # 3. 读取 RGB (Raw)
        rgb_path = os.path.join(rgb_seq_dir, frame_name)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: raise ValueError(f"Failed to load image: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # 转 RGB

        # 4. 读取对应的 Mask (文件名相同，后缀为 .png)
        mask_name = frame_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_root, rel_seq_path, mask_name)
        
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 读取单通道
            # 归一化到 0~1
            mask_img = mask_img.astype(np.float32) / 255.0
            # 扩展维度以匹配 RGB: (H, W) -> (H, W, 1)
            mask_img = mask_img[:, :, np.newaxis]
            
            # === 核心：生成 Masked RGB ===
            # 背景变黑，保留彩色人体
            masked_rgb = (rgb_img * mask_img).astype(np.uint8)
        else:
            # 如果没有 Mask (极少数异常情况)，降级使用 Raw RGB
            masked_rgb = rgb_img
            
        # 5. 转为 PIL Image (以便使用 torchvision transform)
        return Image.fromarray(masked_rgb)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # === 1. 加载图像 (Reference & Target) ===
            ref_image = self._load_random_frame(item['ref']['seq_path'])
            tar_image = self._load_random_frame(item['tar']['seq_path'])
            
            # 应用 CLIP 的预处理
            if self.transform:
                ref_image = self.transform(ref_image)
                tar_image = self.transform(tar_image)
            
            # === 2. 处理文本 (Instruction) ===
            raw_caption = item['caption']
            # 替换占位符: "Add a bag to {subject}." -> "Add a bag to the person."
            caption = raw_caption.replace("{subject}", self.subject_token)
            
            result = {
                "ref_img": ref_image,
                "tar_img": tar_image,
                "text": caption,
                "task": item['task'] # 用于评估时区分任务类型
            }
            
            # === 3. (可选) 处理静态描述 ===
            # 如果你在训练配置里开了 use_aux_loss=True，这里就有用了
            if self.return_static:
                # 替换占位符
                ref_static = item['ref']['static_caption'].replace("{subject}", self.subject_token)
                tar_static = item['tar']['static_caption'].replace("{subject}", self.subject_token)
                
                result["ref_text"] = ref_static
                result["tar_text"] = tar_static
                
            return result
            
        except Exception as e:
            # 简单的错误处理：如果这就坏了，打印错误并随机返回另一个样本
            # 防止训练中断
            print(f"Error loading index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.data)-1))

    def __len__(self):
        return len(self.data)