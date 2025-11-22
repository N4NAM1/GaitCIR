import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class Combiner(nn.Module):
    """
    轻量级融合网络：接收 (Image_Feat, Text_Feat)，输出 Query_Feat
    """
    def __init__(self, input_dim=512, hidden_dim=2048):
        super().__init__()
        self.input_dim = input_dim * 2 
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 残差系数 alpha
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, img_feat, txt_feat):
        # 1. 拼接
        combined = torch.cat((img_feat, txt_feat), dim=-1)
        # 2. 计算 Delta
        delta = self.layers(combined)
        # 3. 残差融合 (注意：这里返回的是未归一化的特征，保留幅度信息)
        return img_feat + self.alpha * delta

class GaitCIRModel(nn.Module):
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_id)
        
        # 冻结 CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.feature_dim = self.clip.projection_dim
        self.combiner = Combiner(self.feature_dim)
        
        # 可学习的 Logit Scale
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def extract_img_feature(self, pixel_values):
        with torch.no_grad():
            feat = self.clip.get_image_features(pixel_values)
        return F.normalize(feat, dim=-1)

    def extract_txt_feature(self, input_ids, attention_mask):
        with torch.no_grad():
            feat = self.clip.get_text_features(input_ids, attention_mask)
        return F.normalize(feat, dim=-1)

    def forward(self, ref_pixels, input_ids, attention_mask):
        """
        默认推理模式 (Forward Retrieval)
        Ref + Text -> Pred
        """
        ref_feat = self.extract_img_feature(ref_pixels)
        txt_feat = self.extract_txt_feature(input_ids, attention_mask)
        
        pred_feat = self.combiner(ref_feat, txt_feat)
        return F.normalize(pred_feat, dim=-1)