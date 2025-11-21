import torch
import torch.nn as nn
from transformers import CLIPModel

class Combiner(nn.Module):
    """
    轻量级融合网络：接收 (Image_Feat, Text_Feat)，输出 Query_Feat
    """
    def __init__(self, input_dim=512, hidden_dim=2048):
        super().__init__()
        # 输入是 图像特征 + 文本特征 拼接
        self.input_dim = input_dim * 2 
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(hidden_dim, input_dim) # 映射回 CLIP 空间
        )
        
        # 学习一个残差系数 alpha (初始为 0，即初始状态下输出 = 原图特征)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, img_feat, txt_feat):
        # 1. 拼接
        combined = torch.cat((img_feat, txt_feat), dim=-1)
        
        # 2. 计算修改量 (Delta)
        delta = self.layers(combined)
        
        # 3. 残差融合
        # Query = Image + alpha * Delta
        return img_feat + self.alpha * delta

class GaitCIRModel(nn.Module):
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        super().__init__()
        # 加载预训练 CLIP
        self.clip = CLIPModel.from_pretrained(model_id)
        
        # 冻结 CLIP 所有参数
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # 初始化 Combiner
        self.feature_dim = self.clip.projection_dim # 512
        self.combiner = Combiner(self.feature_dim)
        
        # 这是一个可学习的温度系数，用于 Loss 计算
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # log(1/0.07)

    def extract_img_feature(self, pixel_values):
        with torch.no_grad():
            feat = self.clip.get_image_features(pixel_values)
        return feat / feat.norm(dim=-1, keepdim=True)

    def extract_txt_feature(self, input_ids, attention_mask):
        with torch.no_grad():
            feat = self.clip.get_text_features(input_ids, attention_mask)
        return feat / feat.norm(dim=-1, keepdim=True)

    def forward(self, ref_pixels, input_ids, attention_mask):
        """训练时的前向传播：计算 Query Embedding"""
        # 1. 提取冻结特征
        ref_feat = self.extract_img_feature(ref_pixels)
        txt_feat = self.extract_txt_feature(input_ids, attention_mask)
        
        # 2. 融合 (这一步有梯度)
        query_feat = self.combiner(ref_feat, txt_feat)
        
        # 归一化，准备算 Cosine Similarity
        return query_feat / query_feat.norm(dim=-1, keepdim=True)