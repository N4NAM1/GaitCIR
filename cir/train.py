import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import CLIPProcessor, AdamW, get_scheduler
from tqdm import tqdm

# 引入你的模块
from data.dataset_loader import GaitCIRDataset
from modeling.demo_model import GaitCIRModel

# ================= 配置 =================
# 路径配置
TRAIN_JSON = '../datasets/GaitCIR_RGB/casiab_cir_train_split.json'
DATA_ROOT = '../datasets/CASIA-B-Processed'
OUTPUT_DIR = './checkpoints'

# 超参数
MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64      # 显存够大可以开到 128
LR = 1e-4            # 仅训练 MLP，学习率可以稍大
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================================

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 准备组件
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = GaitCIRModel(MODEL_ID).to(DEVICE)
    
    # 优化器只优化 Combiner 和 logit_scale
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    
    # Collate Function: 处理 Batch 中的 Tokenization
    def collate_fn(batch):
        ref_imgs = [item['ref_img'] for item in batch]
        tar_imgs = [item['tar_img'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # CLIP 预处理
        inputs = processor(text=texts, images=ref_imgs + tar_imgs, return_tensors="pt", padding=True)
        
        # 拆分图片 Tensor (Batch*2 -> Ref, Tar)
        ref_pix, tar_pix = inputs['pixel_values'].chunk(2)
        
        return ref_pix, tar_pix, inputs['input_ids'], inputs['attention_mask']

    # DataLoader
    dataset = GaitCIRDataset(TRAIN_JSON, DATA_ROOT, transform=None, subject_token="the person")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # Scheduler
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=100, num_training_steps=len(loader)*EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    # 2. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for ref, tar, txt_ids, txt_mask in pbar:
            ref, tar = ref.to(DEVICE), tar.to(DEVICE)
            txt_ids, txt_mask = txt_ids.to(DEVICE), txt_mask.to(DEVICE)
            
            optimizer.zero_grad()
            
            # === Forward ===
            # 1. 计算 Query 特征
            query_feat = model(ref, txt_ids, txt_mask)
            
            # 2. 计算 Target 特征 (作为正样本 Ground Truth)
            # 注意：这里直接用 CLIP 提取 Tar 特征，不经过 Combiner
            with torch.no_grad():
                target_feat = model.extract_img_feature(tar)
            
            # === Compute Loss (Batch-based Contrastive) ===
            # 每一个 Query 应该和同一行里的 Target 相似度最高
            # 相似度矩阵: [B, B]
            logits = (query_feat @ target_feat.T) * model.logit_scale.exp()
            labels = torch.arange(logits.size(0)).to(DEVICE)
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # 保存模型 (只保存 Combiner 权重，很小)
        torch.save(model.combiner.state_dict(), f"{OUTPUT_DIR}/combiner_ep{epoch+1}.pth")

if __name__ == '__main__':
    train()