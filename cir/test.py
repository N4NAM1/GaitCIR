import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
# 设置 HF 镜像 (如果需要的话，建议保留在入口脚本或环境变量中)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import CLIPProcessor

import config as cfg 
from modeling.demo_model import GaitCIRModel
from data.dataset_loader import GaitCIRDataset
from data.collate import get_collate_fn
from utils.Metrics import compute_hierarchical_metrics

# 显式指定测试哪个权重 (方便消融实验对比)
TEST_CHECKPOINT = "./checkpoints/cycle_on/L_cycle_combiner_ep30.pth"

@torch.no_grad()
def evaluate():
    print(f"🚀 Loading Model from: {TEST_CHECKPOINT}")
    
    # 1. 模型加载
    model = GaitCIRModel(cfg.MODEL_ID).to(cfg.DEVICE)
    if os.path.exists(TEST_CHECKPOINT):
        state_dict = torch.load(TEST_CHECKPOINT)
        model.combiner.load_state_dict(state_dict)
        print("✅ 权重加载成功!")
    else:
        print(f"❌ 错误: 找不到权重文件 {TEST_CHECKPOINT}")
        return # 退出

    model.eval()
    processor = CLIPProcessor.from_pretrained(cfg.MODEL_ID)
    
    # 2. 数据加载 (Mode='test')
    # collate_fn 返回: ref_stack, tar_stack, input_ids, attention_mask, tasks, meta
    collate_fn = get_collate_fn(processor, mode='test')

    dataset = GaitCIRDataset(
        json_path=cfg.TRAIN_JSON,       
        data_root=cfg.DATASET_ROOT,
        split_config_path=cfg.SPLIT_CONFIG,
        mode='test',                    
        max_frames=cfg.NUM_FRAMES if hasattr(cfg, 'NUM_FRAMES') else 8, 
        subject_token="the person"
    )

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)

    all_q, all_t = [], []
    all_meta, all_tasks = [], []
    
    print(f"🔍 开始测试 (N={len(dataset)})...")
    
    # 注意：这里变量数量必须和 collate.py 的 test 模式返回一致 (6个)
    for ref, tar, ids, mask, tasks, meta in tqdm(loader):
        B, T, C, H, W = ref.shape
        
        # 展平处理多帧
        ref = ref.view(B*T, C, H, W).to(cfg.DEVICE)
        tar = tar.view(B*T, C, H, W).to(cfg.DEVICE)
        ids = ids.to(cfg.DEVICE)
        mask = mask.to(cfg.DEVICE)
        
        with torch.cuda.amp.autocast():
            # 特征提取
            ref_f = model.extract_img_feature(ref) # [B*T, 512]
            tar_f = model.extract_img_feature(tar)
            
            # 平均池化聚合多帧特征 [B*T, 512] -> [B, 512]
            ref_f = ref_f.view(B, T, -1).mean(dim=1)
            tar_f = tar_f.view(B, T, -1).mean(dim=1)
            
            # 归一化 (Mean 之后需要重新归一化)
            ref_f = torch.nn.functional.normalize(ref_f, dim=-1)
            tar_f = torch.nn.functional.normalize(tar_f, dim=-1)
            
            txt_f = model.extract_txt_feature(ids, mask)
            
            # Combiner 融合
            # 这里直接调用 combiner 得到 raw feature，也可以像 forward 那样再次归一化
            q_f_raw = model.combiner(ref_f, txt_f)
            q_f = torch.nn.functional.normalize(q_f_raw, dim=-1)
        
        all_q.append(q_f.float().cpu())
        all_t.append(tar_f.float().cpu())
        all_tasks.extend(tasks)
        all_meta.extend(meta)
        
    all_q = torch.cat(all_q, dim=0)
    all_t = torch.cat(all_t, dim=0)
    
    # 3. 指标计算
    metrics = compute_hierarchical_metrics(all_q, all_t, all_meta, all_meta, all_tasks)
    
    # 打印结果 (保持原有格式)
    print_metrics(metrics)

def print_metrics(metrics):
    print("\n" + "="*85)
    print(f"{'Task Type':<20} | {'Metric':<8} | {'R@1':<6} | {'R@5':<6} | {'R@10':<6}")
    print("-" * 85)
    order = ["attribute_change", "viewpoint_change", "composite_change", "Overall"]
    for task in order:
        if task in metrics:
            res = metrics[task]
            print(f"{task:<20} ({res['Count']})")
            s = res['Strict']
            print(f"  {'':<20} | {'Strict':<8} | {s['R1']:.1f}   | {s['R5']:.1f}   | {s['R10']:.1f}")
            so = res['Soft']
            print(f"  {'':<20} | {'Soft':<8} | {so['R1']:.1f}   | {so['R5']:.1f}   | -")
            i = res['ID']
            print(f"  {'':<20} | {'ID-Only':<8} | {i['R1']:.1f}   | {i['R5']:.1f}   | -")
            print("-" * 40)
    print("="*85)

if __name__ == '__main__':
    evaluate()