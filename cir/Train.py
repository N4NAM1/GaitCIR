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
from data.collate import get_collate_fn

# ================= 配置 =================
# 路径配置
MASTER_JSON = '../datasets/GaitCIR_RGB/casiab_cir_final.json'
DATA_ROOT = '../datasets/CASIA-B-Processed'
SPLIT_CONFIG = '../datasets/GaitCIR_RGB/Split/CASIA-B.json'
OUTPUT_DIR = './checkpoints/simpleMLP'

# 超参数
# 训练超参
MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64      # 显存允许的话越大越好
LR = 1e-4            
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8      
# ===========================================

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 准备模型
    print(f"🚀 初始化模型: {MODEL_ID}")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = GaitCIRModel(MODEL_ID).to(DEVICE)
    
    # 优化器：只训练 Combiner 和 Logit Scale
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # 2. 准备数据 (带 Split 过滤)
    print("加载训练集...")
    dataset = GaitCIRDataset(
        json_path=MASTER_JSON, 
        data_root=DATA_ROOT, 
        split_config_path=SPLIT_CONFIG, # 【关键】只读取 001-074
        mode='train', 
        max_frames=1, # 随机单帧
        subject_token="the person"
    )

    collate_fn = get_collate_fn(processor, mode='train')

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=100, num_training_steps=len(loader)*EPOCHS)

    # 3. 训练循环
    print(f"开始训练... (总步数: {len(loader)*EPOCHS})")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for ref, tar, txt_ids, txt_mask in pbar:
            ref, tar = ref.to(DEVICE), tar.to(DEVICE)
            txt_ids, txt_mask = txt_ids.to(DEVICE), txt_mask.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            query_feat = model(ref, txt_ids, txt_mask)
            
            with torch.no_grad():
                target_feat = model.extract_img_feature(tar)
            
            # Contrastive Loss
            logit_scale = model.logit_scale.exp()
            logits = (query_feat @ target_feat.T) * logit_scale
            labels = torch.arange(logits.size(0)).to(DEVICE)
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # 保存模型
        save_path = f"{OUTPUT_DIR}/combiner_ep{epoch+1}.pth"
        torch.save(model.combiner.state_dict(), save_path)

if __name__ == '__main__':
    train()