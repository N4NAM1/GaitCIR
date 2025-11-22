import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
# 设置 HF 镜像 (如果需要的话，建议保留在入口脚本或环境变量中)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from tqdm import tqdm
from transformers import CLIPProcessor, AdamW, get_scheduler

# === 引入配置模块 ===
import config as cfg

# 引入你的其他模块
from data.dataset_loader import GaitCIRDataset
from modeling.demo_model2 import GaitCIRModel
from data.collate import get_collate_fn

# ================= 本地训练特定超参 =================
# 这些参数通常随实验变化，可以保留在此处，也可以移入 config
LR = 1e-4            
EPOCHS = 30

# === 【消融实验配置】 ===
# 修改此处可快速进行对比实验
ABLATION_CONFIG = {
    "USE_CYCLE_LOSS": True,  # 开关：是否使用循环一致性 Loss
    "CYCLE_LAMBDA": 1.0,     # 权重：L_total = L_cir + lambda * L_cycle
}
# ===================================================

def train():
    # 使用 config 中的输出路径
    # 为了区分不同实验，建议在输出路径中带上是否使用 Cycle 的标记
    exp_sub_dir = "cycle_on" if ABLATION_CONFIG["USE_CYCLE_LOSS"] else "cycle_off"
    save_dir = os.path.join(cfg.OUTPUT_DIR, exp_sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 实验配置: {ABLATION_CONFIG}")
    print(f"📂 Checkpoints 将保存至: {save_dir}")

    # 1. 准备模型
    print(f"🚀 初始化模型: {cfg.MODEL_ID}")
    processor = CLIPProcessor.from_pretrained(cfg.MODEL_ID)
    model = GaitCIRModel(cfg.MODEL_ID).to(cfg.DEVICE)
    
    # 优化器：只训练 Combiner 和 Logit Scale
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    
    # Loss 定义
    loss_cir_fn = nn.CrossEntropyLoss()       # 用于正向检索 (InfoNCE)
    loss_cycle_fn = nn.CosineEmbeddingLoss()  # 用于循环重构 (Recon -> Ref)

    # 2. 准备数据
    print(f"加载训练集 (Batch Size: {cfg.BATCH_SIZE})...")
    dataset = GaitCIRDataset(
        json_path=cfg.TRAIN_JSON,       # 使用 config 中的路径
        data_root=cfg.DATASET_ROOT,    
        split_config_path=cfg.SPLIT_CONFIG, 
        mode='train', 
        max_frames=1, 
        subject_token="the person"
    )

    # 获取训练模式的 collate_fn (会返回 text_inv)
    collate_fn = get_collate_fn(processor, mode='train')

    loader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=100, num_training_steps=len(loader)*EPOCHS)

    # 3. 训练循环
    print(f"开始训练... (总步数: {len(loader)*EPOCHS}) | 设备: {cfg.DEVICE}")
    
    for epoch in range(EPOCHS):
        model.train()
        
        # 统计变量
        total_loss_avg = 0
        cir_loss_avg = 0
        cycle_loss_avg = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # 注意：这里解包 6 个变量 (适配修改后的 collate.py)
        for ref_imgs, tar_imgs, fwd_ids, fwd_mask, inv_ids, inv_mask in pbar:
            
            # 转移数据到设备
            ref_imgs = ref_imgs.to(cfg.DEVICE)
            tar_imgs = tar_imgs.to(cfg.DEVICE)
            fwd_ids = fwd_ids.to(cfg.DEVICE)
            fwd_mask = fwd_mask.to(cfg.DEVICE)
            inv_ids = inv_ids.to(cfg.DEVICE)
            inv_mask = inv_mask.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            
            # === A. 特征提取 (Frozen CLIP) ===
            # ref_feat: [B, 512] (已归一化)
            ref_feat = model.extract_img_feature(ref_imgs)
            
            with torch.no_grad():
                # tar_feat: [B, 512] (已归一化)
                tar_feat = model.extract_img_feature(tar_imgs)
            
            # fwd_txt_feat: [B, 512] (已归一化)
            fwd_txt_feat = model.extract_txt_feature(fwd_ids, fwd_mask)
            
            # === B. 正向检索过程 (L_cir) ===
            # Ref + T_fwd -> Pred (Combiner 内部输出未归一化特征，这里手动归一化)
            pred_feat_raw = model.combiner(ref_feat, fwd_txt_feat)
            pred_feat = F.normalize(pred_feat_raw, dim=-1) 
            
            # 计算对比损失 (Batch-based classification)
            logit_scale = model.logit_scale.exp()
            logits = (pred_feat @ tar_feat.T) * logit_scale
            labels = torch.arange(logits.size(0)).to(cfg.DEVICE)
            
            l_cir = loss_cir_fn(logits, labels)
            
            # === C. 循环一致性过程 (L_cycle) ===
            l_cycle = torch.tensor(0.0).to(cfg.DEVICE)
            
            if ABLATION_CONFIG["USE_CYCLE_LOSS"]:
                inv_txt_feat = model.extract_txt_feature(inv_ids, inv_mask)
                
                # Pred (Normalized) + T_inv -> Recon
                # 我们希望 Recon 能够重构出原始的 Ref 特征
                recon_feat_raw = model.combiner(pred_feat, inv_txt_feat)
                
                # Cosine Embedding Loss: 
                # Input1: recon, Input2: ref, Target: 1 (表示希望它们相似)
                target_ones = torch.ones(ref_feat.size(0)).to(cfg.DEVICE)
                l_cycle = loss_cycle_fn(recon_feat_raw, ref_feat, target_ones)

            # === D. 总损失 ===
            loss = l_cir + ABLATION_CONFIG["CYCLE_LAMBDA"] * l_cycle
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 记录数据
            total_loss_avg += loss.item()
            cir_loss_avg += l_cir.item()
            cycle_loss_avg += l_cycle.item() if ABLATION_CONFIG["USE_CYCLE_LOSS"] else 0
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "L_cir": f"{l_cir.item():.4f}",
                "L_cyc": f"{l_cycle.item():.4f}"
            })
        
        # Epoch 结束统计
        steps = len(loader)
        avg_loss = total_loss_avg / steps
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f} "
              f"(CIR: {cir_loss_avg/steps:.4f}, Cycle: {cycle_loss_avg/steps:.4f})")
        
        # 保存模型
        save_name = f"L_cycle_combiner_ep{epoch+1}.pth"
        save_path = os.path.join(save_dir, save_name)
        torch.save(model.combiner.state_dict(), save_path)
        print(f"✅ 模型已保存: {save_path}")

if __name__ == '__main__':
    train()