import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        # 1. 特征投影层 (将 CLIP 特征投影到 projection_dim)
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        # 2. 融合层 (输入为 2 * projection_dim)
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim) # 输出回原尺寸

        self.dropout3 = nn.Dropout(0.5)
        
        # 3. 动态标量（Dynamic Scalar）用于残差连接的权重
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 温度系数，这里保留原值，但通常在 GaitCIRModel 中处理
        self.logit_scale = 100 

    # ⚠️ 关键修改 1: 将 combine_features 改为 forward
    def forward(self, image_features, text_features):
        """
        Cobmine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features (agg_feat)
        :param text_features: CLIP relative caption features (txt_feat)
        :return: predicted features (query_feat)
        """
        # 投影与激活
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        # 拼接投影后的特征
        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        
        # 融合 MLP
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        
        # 计算动态标量 sigma
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        
        # 最终融合：输出 MLP + 动态加权残差连接
        # Query = MLP_output + sigma * Txt_feat + (1 - sigma) * Img_feat
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                        1 - dynamic_scalar) * image_features
                        
        # 归一化 (Combiner 内部完成)
        return F.normalize(output)

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
        
        # ⚠️ 关键修改 2: 传入新的 Combiner 所需的三个参数
        self.combiner = Combiner(
            clip_feature_dim=self.feature_dim, 
            projection_dim=self.feature_dim, 
            hidden_dim=2048 # 沿用原来的 2048
        )
        
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
        agg_feat = features.max(dim=1)[0] # -> [B, 512]
        
        # 4. ⚠️ 再次归一化 (Pooling 后模长会变，必须 Re-Norm)
        return F.normalize(agg_feat, dim=-1)

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
        # 融合器内部已经包含了归一化操作
        query_feat = self.combiner(ref_agg, txt_feat)
        
        # 输出归一化 (已移除，因为 Combiner 内部已完成)
        return query_feat