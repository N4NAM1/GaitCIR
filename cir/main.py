import os
# 设置 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

# === 引入项目模块 ===
import config as cfg
from modeling.demo_model import GaitCIRModel
from data.dataset_loader import GaitCIRDataset
from data.collate import get_collate_fn
from utils.Metrics import compute_hierarchical_metrics

# === 环境配置：防止 OpenCV 多线程死锁 ===
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_parser():
    """ 命令行参数解析 """
    parser = argparse.ArgumentParser(description='GaitCIR Main Program')
    
    # === DDP 必要参数 ===
    parser.add_argument('--local_rank', type=int, default=0, help="DDP Local Rank")
    parser.add_argument('--local-rank', type=int, default=0, help="Torch launch compatibility")
    
    # === 基础运行参数 ===
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help="Run mode")
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument('--gpu', default='0,1,2,3', type=str, help="Visible GPUs (info only)")
    
    # === 动态覆盖 Config ===
    parser.add_argument('--no_feat', action='store_true', help="Force Image Mode (Raw RGB)")
    parser.add_argument('--unmasked', action='store_true', help="Force Unmasked Features")
    parser.add_argument('--ckpt', default=None, type=str, help="Checkpoint path for testing")
    
    return parser


def initialization(args):
    """
    环境初始化：DDP 连接、随机种子、配置更新
    """
    # === 1. 自动检测并补全 DDP 环境变量 (兼容直接运行 python main.py) ===
    if 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
        print("⚠️ [Init] No DDP environment found. Falling back to Single-GPU Mode.")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345' # 默认端口
        os.environ['LOCAL_RANK'] = '0'
    
    # === 2. DDP 初始化 ===
    # 优先使用环境变量中的 LOCAL_RANK
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    # 此时 os.environ 中一定有 LOCAL_RANK (要么是 torchrun 设的，要么是我们上面补全的)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # 更新全局设备配置
    cfg.DEVICE = torch.device("cuda", args.local_rank)

    # === 3. 随机种子 ===
    seed = args.seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # === 4. 动态更新 Config ===
    if args.no_feat:
        cfg.USE_FEATURES = False
        if dist.get_rank() == 0: print("⚠️ [Config] Override: Forced Image Mode (Raw RGB)")
    
    if args.unmasked:
        cfg.FEATURE_ROOT = cfg.FEATURE_ROOT_UNMASKED
        cfg.USE_MASK = False
        if dist.get_rank() == 0: print("⚠️ [Config] Override: Using UNMASKED Data")
    
    # === 5. 打印信息 ===
    if dist.get_rank() == 0:
        print(f"🚀 [Init] DDP Initialized. World Size: {dist.get_world_size()}")
        print(f"🚀 [Init] Phase: {args.phase} | Feature Mode: {cfg.USE_FEATURES} | Mask: {cfg.USE_MASK}")
        print(f"🚀 [Loss] Inv Type: {cfg.LOSS_INV_TYPE} | Alpha: {cfg.LOSS_ALPHA}")


def run_model(args):
    """ 模型构建与引擎分发 """
    # 1. 构建模型
    if dist.get_rank() == 0: print(f"🏗️ [Model] Building Backbone: {cfg.MODEL_ID}")
    
    model = GaitCIRModel(cfg.MODEL_ID).to(cfg.DEVICE)
    processor = CLIPProcessor.from_pretrained(cfg.MODEL_ID)
    
    # 2. 加载权重
    if args.phase == 'test':
        if args.ckpt is None:
            raise ValueError("❌ [Error] --ckpt is required for testing phase!")
        if dist.get_rank() == 0: print(f"📥 [Model] Loading Checkpoint: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location=cfg.DEVICE)
        model.combiner.load_state_dict(state_dict)

    # 3. DDP 封装
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.local_rank], 
        output_device=args.local_rank,
        find_unused_parameters=True 
    )

    # 4. 启动引擎
    if args.phase == 'train':
        train_engine(model, processor, args)
    else:
        test_engine(model, processor, args)


def train_engine(model, processor, args):
    """ 训练引擎 """
    # === 1. 数据准备 ===
    dataset = GaitCIRDataset(
        json_path=cfg.TRAIN_JSON, 
        data_root=cfg.DATASET_ROOT, 
        split_config_path=cfg.SPLIT_CONFIG,
        dataset_name=cfg.DATASET_NAME, # 🔥 [修正] 必传参数
        mode='train', 
        max_frames=cfg.TRAIN_MAX_FRAMES, 
        use_features=cfg.USE_FEATURES,
        feature_root=cfg.FEATURE_ROOT,
        use_mask=cfg.USE_MASK
    )
    
    sampler = DistributedSampler(dataset, shuffle=True)
    collate_fn = get_collate_fn(processor, mode='train')
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    
    # === 2. 优化器 ===
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    cosine_loss_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y).mean()
    
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=cfg.WARMUP_STEPS, 
                              num_training_steps=len(loader) * cfg.EPOCHS)

    # === 3. 训练循环 ===
    if dist.get_rank() == 0:
        print(f"🚀 [Engine] Start Training ({cfg.EPOCHS} Epochs)...")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
    model.train()
    
    for epoch in range(cfg.EPOCHS):
        sampler.set_epoch(epoch)
        
        total_loss = 0
        
        header = f"🚀 Train Ep {epoch+1}/{cfg.EPOCHS}"
        iterator = tqdm(loader, desc=header) if dist.get_rank() == 0 else loader
        
        for batch in iterator:
            if batch is None: continue
            
            ref, tar, txt_ids, txt_mask, inv_ids, inv_mask = batch
            
            txt_ids, txt_mask = txt_ids.to(cfg.DEVICE), txt_mask.to(cfg.DEVICE)
            inv_ids, inv_mask = inv_ids.to(cfg.DEVICE), inv_mask.to(cfg.DEVICE)
            
            if cfg.USE_FEATURES:
                ref, tar = ref.to(cfg.DEVICE), tar.to(cfg.DEVICE)
            else:
                B, T, C, H, W = ref.shape
                ref = ref.view(-1, C, H, W).to(cfg.DEVICE) 
                tar = tar.view(-1, C, H, W).to(cfg.DEVICE)
                
            optimizer.zero_grad()
            
            # Forward
            raw_model = model.module
            
            if cfg.USE_FEATURES:
                ref_agg = raw_model.aggregate_features(ref, ref.size(0), ref.size(1))
                tar_agg = raw_model.aggregate_features(tar, tar.size(0), tar.size(1))
            else:
                ref_feat = raw_model.extract_img_feature(ref)
                tar_feat = raw_model.extract_img_feature(tar)
                ref_agg = raw_model.aggregate_features(ref_feat, B, T)
                tar_agg = raw_model.aggregate_features(tar_feat, B, T)

            txt_feat = raw_model.extract_txt_feature(txt_ids, txt_mask)
            inv_feat = raw_model.extract_txt_feature(inv_ids, inv_mask)
            
            q_fwd = raw_model.combiner(ref_agg, txt_feat) 
            q_inv = raw_model.combiner(tar_agg, inv_feat) 
            
            # Loss
            logit_scale = raw_model.logit_scale.exp()
            labels = torch.arange(len(q_fwd)).to(cfg.DEVICE)
            
            logits_fwd = (q_fwd @ tar_agg.T) * logit_scale
            loss_fwd = loss_fn(logits_fwd, labels)
            
            if cfg.LOSS_INV_TYPE == 'nce':
                logits_inv = (q_inv @ ref_agg.T) * logit_scale
                loss_inv = loss_fn(logits_inv, labels)
            else:
                loss_inv = cosine_loss_fn(q_inv, ref_agg)
            
            loss = loss_fwd + cfg.LOSS_ALPHA * loss_inv
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if dist.get_rank() == 0:
                current_lr = optimizer.param_groups[0]['lr']
                iterator.set_postfix({
                    "L": f"{loss.item():.4f}", 
                    "Lf": f"{loss_fwd.item():.3f}", 
                    "Li": f"{loss_inv.item():.3f}",
                    "LR": f"{current_lr:.1e}"
                })
        
        # Epoch End
        if dist.get_rank() == 0:
            avg_loss = total_loss / len(loader)
            print(f"✅ Epoch {epoch+1}/{cfg.EPOCHS} Done. Avg Loss: {avg_loss:.4f}")
            
            save_path = os.path.join(cfg.OUTPUT_DIR, f"combiner_ep{epoch+1}.pth")
            torch.save(model.module.combiner.state_dict(), save_path)

def print_report(metrics):
    """
    打印详细的测试报告表格 (仿 Test.py 风格 + mAP)
    """
    print("\n" + "="*95)
    # 表头增加 mAP
    print(f"{'Task Type':<20} | {'Metric':<8} | {'R@1':<6} | {'R@5':<6} | {'R@10':<6} | {'mAP':<6}")
    print("-" * 95)
    
    # 定义任务显示顺序
    order = ["attribute_change", "viewpoint_change", "Overall"]
    
    for task in order:
        if task in metrics:
            res = metrics[task]
            count = res['Count']
            print(f"{task:<20} (N={count})")
            
            # 1. Strict 指标 (最重要)
            s = res['Strict']
            print(f"  {'':<20} | {'Strict':<8} | {s['R1']:>6.1f} | {s['R5']:>6.1f} | {s['R10']:>6.1f} | {s['mAP']:>6.1f}")
            
            # 2. Soft 指标 (宽松匹配)
            if 'Soft' in res:
                so = res['Soft']
                print(f"  {'':<20} | {'Soft':<8} | {so['R1']:>6.1f} | {so['R5']:>6.1f} | {so['R10']:>6.1f} | {so['mAP']:>6.1f}")
            
            # 3. ID 指标 (是否找对了人)
            if 'ID' in res:
                i = res['ID']
                print(f"  {'':<20} | {'ID-Only':<8} | {i['R1']:>6.1f} | {i['R5']:>6.1f} | {i['R10']:>6.1f} | {i['mAP']:>6.1f}")
            
            print("-" * 40)
            
    print("="*95 + "\n")


@torch.no_grad()
def test_engine(model, processor, args):
    """ 测试引擎 """
    if dist.get_rank() == 0: print("🔍 [Engine] Start Testing...")
    model.eval()
    
    dataset = GaitCIRDataset(
        json_path=cfg.TRAIN_JSON, 
        data_root=cfg.DATASET_ROOT, 
        split_config_path=cfg.SPLIT_CONFIG,
        dataset_name=cfg.DATASET_NAME, # 🔥 [修正] 必传参数，用于处理不同数据集路径逻辑
        mode='test', 
        max_frames=cfg.TEST_MAX_FRAMES, 
        use_features=cfg.USE_FEATURES,
        feature_root=cfg.FEATURE_ROOT,
        use_mask=cfg.USE_MASK
    )
    
    sampler = DistributedSampler(dataset, shuffle=False)
    collate_fn = get_collate_fn(processor, mode='test')
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, 
                        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)
    
    all_q, all_t = [], []
    all_meta, all_tasks = [], []
    
    iterator = tqdm(loader, desc="🔍 Testing") if dist.get_rank() == 0 else loader
    
    for batch in iterator:
        if batch is None: continue
        ref, tar, ids, mask, tasks, meta = batch
        
        # 1. 文本数据
        ids, mask = ids.to(cfg.DEVICE), mask.to(cfg.DEVICE)
        
        # 获取原始模型 (解开 DDP 包装)
        raw_model = model.module if hasattr(model, 'module') else model
        
        # 2. 视觉数据处理 (兼容 Feature / Image)
        if cfg.USE_FEATURES:
            # 特征模式: [T, 512]
            if isinstance(ref, list):
                ref_agg_list = []
                tar_agg_list = []
                
                # 逐个样本处理 (处理变长序列)
                for r, t in zip(ref, tar):
                    r = r.to(cfg.DEVICE)
                    t = t.to(cfg.DEVICE)
                    
                    # 聚合: [T, 512] -> [1, T, 512] -> [1, 512]
                    r_agg = raw_model.aggregate_features(r.unsqueeze(0), 1, r.size(0))
                    t_agg = raw_model.aggregate_features(t.unsqueeze(0), 1, t.size(0))
                    
                    ref_agg_list.append(r_agg)
                    tar_agg_list.append(t_agg)
                
                # 重新堆叠为 Batch [B, 512]
                ref_agg = torch.cat(ref_agg_list, dim=0)
                tar_agg = torch.cat(tar_agg_list, dim=0)
            else:
                # 兜底：如果是 Tensor
                ref, tar = ref.to(cfg.DEVICE), tar.to(cfg.DEVICE)
                ref_agg = raw_model.aggregate_features(ref, ref.size(0), ref.size(1))
                tar_agg = raw_model.aggregate_features(tar, tar.size(0), tar.size(1))
        else:
            # 🔥 [修正] Image Mode (Raw RGB): [B, T, C, H, W]
            ref = ref.to(cfg.DEVICE)
            tar = tar.to(cfg.DEVICE)
            
            # 处理 Reference
            B_r, T_r, C, H, W = ref.shape
            ref_flat = ref.view(-1, C, H, W)
            ref_feat = raw_model.extract_img_feature(ref_flat) # [B*T, 512]
            ref_agg = raw_model.aggregate_features(ref_feat, B_r, T_r) # [B, 512]
            
            # 处理 Target
            B_t, T_t, _, _, _ = tar.shape
            tar_flat = tar.view(-1, C, H, W)
            tar_feat = raw_model.extract_img_feature(tar_flat)
            tar_agg = raw_model.aggregate_features(tar_feat, B_t, T_t)

        # 3. 文本特征
        txt_f = raw_model.extract_txt_feature(ids, mask)
        
        # 4. 融合
        q_f = raw_model.combiner(ref_agg, txt_f)
        
        all_q.append(q_f.cpu())
        all_t.append(tar_agg.cpu())
        all_tasks.extend(tasks)
        all_meta.extend(meta)
        
    all_q = torch.cat(all_q, dim=0)
    all_t = torch.cat(all_t, dim=0)
    
    if dist.get_rank() == 0:
        print(f"📊 Computing Metrics (Rank 0 Data Size: {len(all_q)})...")
        
        # 计算指标
        metrics = compute_hierarchical_metrics(all_q, all_t, all_meta, all_meta, all_tasks)
        
        # 打印报表
        print_report(metrics)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    initialization(args)
    run_model(args)