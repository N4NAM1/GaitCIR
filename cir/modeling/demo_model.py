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
        
        # 残差系数
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, img_feat, txt_feat):
        combined = torch.cat((img_feat, txt_feat), dim=-1)
        delta = self.layers(combined)
        # Residual Connection
        output = img_feat + self.alpha * delta
        return F.normalize(output, dim=-1)

class GaitCIRModel(nn.Module):
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        super().__init__()
        print(f"Loading CLIP: {model_id}...")
        self.clip = CLIPModel.from_pretrained(model_id)
        
        # ❄️ 冻结 CLIP 视觉和文本部分，只训练 Combiner
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # 初始化组件
        self.feature_dim = self.clip.projection_dim # 512
        self.combiner = Combiner(self.feature_dim)
        
        # 可学习的温度系数
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def extract_img_feature(self, pixel_values):
        """ 单帧特征提取: [N, 3, H, W] -> [N, 512] """
        with torch.no_grad():
            feat = self.clip.get_image_features(pixel_values)
        return F.normalize(feat, dim=-1)

    def extract_txt_feature(self, input_ids, attention_mask):
        """ 文本特征提取: [B, L] -> [B, 512] """
        with torch.no_grad():
            feat = self.clip.get_text_features(input_ids, attention_mask)
        return F.normalize(feat, dim=-1)

    def aggregate_features(self, inputs, batch_size, frames_num):
        """
        🔥 核心升级：GaitSet 风格的时序聚合 (Set Pooling)
        支持输入 'Image Tensor' 或 'Feature Tensor'，自动处理。
        """
        # 1. 如果输入是图片 [B*T, 3, H, W]，先提取特征
        if inputs.dim() == 4:
            features = self.extract_img_feature(inputs) # -> [B*T, 512]
        else:
            features = inputs # 已经是 [B*T, 512] 或 [B, T, 512]

        # 2. 统一维度 -> [B, T, D]
        if features.dim() == 2:
            features = features.view(batch_size, frames_num, -1)
            
        # 3. Max Pooling (GaitSet 也是用的 Max)
        # max() 返回 (values, indices)，我们需要 values
        agg_feat = features.max(dim=1)[0] # -> [B, 512]
        
        # 4. ⚠️ 再次归一化 (Pooling 后模长会变，必须 Re-Norm)
        agg_feat = F.normalize(agg_feat, dim=-1)
        
        return agg_feat

    def forward(self, ref_input, input_ids, attention_mask):
        """
        训练前向传播：计算 Query Embedding
        自动判断 ref_input 是图片还是特征
        """
        batch_size = input_ids.size(0)
        
        # === 1. 视觉处理 (Extract + Aggregate) ===
        if ref_input.dim() == 4:
            # Image Mode: [N, 3, H, W] -> 需要计算 T
            total_imgs = ref_input.size(0)
            frames_num = total_imgs // batch_size
            ref_agg = self.aggregate_features(ref_input, batch_size, frames_num)
            
        elif ref_input.dim() == 3:
            # Feature Mode: [B, T, D]
            frames_num = ref_input.size(1)
            ref_agg = self.aggregate_features(ref_input, batch_size, frames_num)
            
        else:
            raise ValueError(f"Unknown input shape: {ref_input.shape}")

        # === 2. 文本处理 ===
        txt_feat = self.extract_txt_feature(input_ids, attention_mask)
        
        # === 3. 融合 (Ref + Text -> Query) ===
        query_feat = self.combiner(ref_agg, txt_feat)
        
        # 输出归一化
        return query_feat