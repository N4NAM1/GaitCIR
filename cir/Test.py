import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

# 1. 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 这里的 cv2 设置建议放在 dataset_loader 或 __init__ 里，不过放在这也行
from transformers import CLIPProcessor
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# 2. 引入核心模块
import config as cfg  # 引入全局配置 (推荐)
from modeling.demo_model import GaitCIRModel
from data.dataset_loader import GaitCIRDataset
from utils.Metrics import compute_hierarchical_metrics
from data.collate import get_collate_fn

# ================= 本地配置 (如果不使用 cfg，可取消注释覆盖) =================
# TEST_JSON = './datasets/GaitCIR_RGB/casiab_cir_test_split.json'
# SPLIT_CONFIG = './datasets/GaitCIR_RGB/casiab_split_config.json'
# DATA_ROOT = './datasets/CASIA-B-Processed'
# CHECKPOINT = './outputs/checkpoints/combiner_ep15.pth' 
# BATCH_SIZE = 32
# NUM_FRAMES = 8 
# NUM_WORKERS = 4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================================

@torch.no_grad()
def evaluate():
    # 使用 config 中的变量 (如果没用 config.py，请替换为上面的本地变量)
    checkpoint_path = cfg.OUTPUT_DIR + "/combiner_ep15.pth" # 或者直接用上面的 CHECKPOINT
    
    print(f"🚀 Loading Model from: {checkpoint_path}")
    model = GaitCIRModel(cfg.MODEL_ID).to(cfg.DEVICE)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model.combiner.load_state_dict(state_dict)
        print("✅ 权重加载成功!")
    else:
        print(f"❌ 警告: 找不到权重文件，将使用随机初始化测试！")
        
    model.eval()

    # 准备处理器
    processor = CLIPProcessor.from_pretrained(cfg.MODEL_ID)
    
    # 获取测试专用的 Collate Fn (支持 List[Tensor] 堆叠和元数据传递)
    collate_fn = get_collate_fn(processor, mode='test')

    # 初始化 Dataset
    print(f"Loading Test Dataset...")
    dataset = GaitCIRDataset(
        json_path=cfg.TRAIN_JSON,       # 注意：这里通常传入 Master JSON
        data_root=cfg.DATASET_ROOT,
        split_config_path=cfg.SPLIT_CONFIG,
        mode='test',                    # 指定测试模式
        max_frames=cfg.NUM_FRAMES if hasattr(cfg, 'NUM_FRAMES') else 8, # 测试时采样多帧
        subject_token="the person"
    )

    loader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS, 
        collate_fn=collate_fn
    )

    all_q, all_t = [], []
    all_meta, all_tasks = [], []
    
    print(f"🔍 开始特征提取 (Test Set Size: {len(dataset)})...")
    
    for ref, tar, ids, mask, tasks, meta in tqdm(loader):
        # Ref/Tar Shape: [B, T, C, H, W]
        B, T, C, H, W = ref.shape
        
        ref = ref.view(B*T, C, H, W).to(cfg.DEVICE)
        tar = tar.view(B*T, C, H, W).to(cfg.DEVICE)
        ids = ids.to(cfg.DEVICE)
        mask = mask.to(cfg.DEVICE)
        
        with torch.cuda.amp.autocast():
            # 1. 提取 Frame 特征
            ref_f = model.extract_img_feature(ref).view(B, T, -1).mean(dim=1)
            tar_f = model.extract_img_feature(tar).view(B, T, -1).mean(dim=1)
            
            # 2. 提取文本
            txt_f = model.extract_txt_feature(ids, mask)
            
            # 3. 融合
            q_f = model.combiner(ref_f, txt_f)
        
        all_q.append(q_f.float().cpu())
        all_t.append(tar_f.float().cpu())
        all_tasks.extend(tasks)
        all_meta.extend(meta)
        
    all_q = torch.cat(all_q, dim=0)
    all_t = torch.cat(all_t, dim=0)
    
    print(f"✅ 特征提取完成。Query Shape: {all_q.shape}")
    
    # 计算分层指标 (调用 utils)
    metrics = compute_hierarchical_metrics(all_q, all_t, all_meta, all_meta, all_tasks)
    
    # 打印报表
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