import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
# 1. 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from transformers import CLIPProcessor



from modeling.demo_model import GaitCIRModel 

# ================= 配置区域 =================
TEST_JSON = '../datasets/GaitCIR_RGB/casiab_cir_test_split.json'
DATA_ROOT = '../datasets/CASIA-B-Processed'
CHECKPOINT = './checkpoints/combiner_ep15.pth' 

BATCH_SIZE = 32
NUM_FRAMES = 8 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8 
# ===========================================

class TestGaitDataset(Dataset):
    """测试集 Loader：返回图像张量及详细元数据"""
    def __init__(self, json_path, data_root, num_frames=8, subject_token="the person"):
        print(f"加载测试集: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.rgb_root = os.path.join(data_root, 'RGB')
        self.mask_root = os.path.join(data_root, 'Mask')
        self.num_frames = num_frames
        self.subject_token = subject_token
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 简单缓存文件列表
        self.seq_cache = {}
        unique_seqs = set()
        for item in self.data:
            unique_seqs.add(item['ref']['seq_path'])
            unique_seqs.add(item['tar']['seq_path'])
        print("正在缓存文件列表...")
        for seq in tqdm(unique_seqs):
            p = os.path.join(self.rgb_root, seq)
            if os.path.isdir(p):
                self.seq_cache[seq] = sorted([f for f in os.listdir(p) if f.endswith('.jpg')])
            else:
                self.seq_cache[seq] = []

    def _load_processed_tensor(self, rel_seq_path):
        all_frames = self.seq_cache.get(rel_seq_path, [])
        total = len(all_frames)
        if total == 0: return torch.zeros(self.num_frames, 3, 224, 224)

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        pil_imgs = []
        base_dir = os.path.join(self.rgb_root, rel_seq_path)

        for idx in indices:
            path = os.path.join(base_dir, all_frames[idx])
            img = cv2.imread(path)
            if img is None: 
                pil_imgs.append(Image.new('RGB', (224, 224)))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_imgs.append(Image.fromarray(img))
        
        while len(pil_imgs) < self.num_frames:
            pil_imgs.append(Image.new('RGB', (224, 224)))

        return self.processor(images=pil_imgs, return_tensors="pt")['pixel_values']

    def __getitem__(self, idx):
        item = self.data[idx]
        ref_t = self._load_processed_tensor(item['ref']['seq_path'])
        tar_t = self._load_processed_tensor(item['tar']['seq_path'])
        
        cap = item['caption'].replace("{subject}", self.subject_token)
        txt = self.processor(text=cap, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        
        return {
            "ref_imgs": ref_t, 
            "tar_imgs": tar_t,
            "input_ids": txt['input_ids'].squeeze(0),
            "attention_mask": txt['attention_mask'].squeeze(0),
            "task": item['task'],
            # === 关键元数据 ===
            "sid": str(item['sid']),
            "cond": str(item['tar']['condition']),
            "view": str(item['tar']['view'])
        }

    def __len__(self): return len(self.data)

def compute_hierarchical_metrics(query_feats, gallery_feats, q_meta, g_meta, tasks):
    """
    计算分层指标 (Hierarchical Metrics)
    """
    print("🚀 正在计算分层相似度矩阵 (CPU)...")
    # 转 CPU 避免 OOM
    query_feats = F.normalize(query_feats, dim=-1).cpu()
    gallery_feats = F.normalize(gallery_feats, dim=-1).cpu()
    
    # 1. 相似度与排序
    sim_matrix = query_feats @ gallery_feats.T
    _, indices = torch.sort(sim_matrix, dim=1, descending=True)
    
    # 2. 准备广播用的属性向量
    print("构建真值矩阵...")
    q_sid = np.array([m['sid'] for m in q_meta])
    g_sid = np.array([m['sid'] for m in g_meta])
    
    q_cond = np.array([m['cond'] for m in q_meta])
    g_cond = np.array([m['cond'] for m in g_meta])
    
    q_view = np.array([m['view'] for m in q_meta])
    g_view = np.array([m['view'] for m in g_meta])
    
    # 3. 构建三个层级的 Ground Truth 矩阵 [N, N]
    # Level 1: Identity Match
    gt_id = (q_sid[:, None] == g_sid[None, :])
    
    # Level 2: Strict Match (ID + Cond + View) [主指标]
    gt_strict = gt_id & (q_cond[:, None] == g_cond[None, :]) & (q_view[:, None] == g_view[None, :])
    
    # Level 3: Soft Match (ID + Cond) [辅助指标]
    gt_soft = gt_id & (q_cond[:, None] == g_cond[None, :])
    
    # 转换为 Tensor
    gt_id_t = torch.from_numpy(gt_id)
    gt_strict_t = torch.from_numpy(gt_strict)
    gt_soft_t = torch.from_numpy(gt_soft)
    
    # 4. 定义计算函数
    def calc_recall(gt_matrix, topk_indices, mask):
        # 筛选当前任务的行
        indices_masked = topk_indices[mask]
        gt_masked = gt_matrix[mask]
        
        if gt_masked.size(0) == 0: return 0.0, 0.0, 0.0
        
        # gather: 在 GT 矩阵中查预测出的索引是否为 True
        hits = torch.gather(gt_masked, 1, indices_masked)
        
        r1 = hits[:, :1].any(dim=1).float().mean().item() * 100
        r5 = hits[:, :5].any(dim=1).float().mean().item() * 100
        r10 = hits[:, :10].any(dim=1).float().mean().item() * 100
        return r1, r5, r10

    # 5. 分任务、分层级统计
    results = {}
    unique_tasks = set(tasks)
    unique_tasks.add("Overall")
    
    # 预先取好 Top-10 索引
    top10_indices = indices[:, :10]
    
    tasks_tensor = np.array(tasks) # 用于做 mask
    
    for task_name in unique_tasks:
        if task_name == "Overall":
            mask = torch.ones(len(tasks), dtype=torch.bool)
        else:
            mask = torch.from_numpy(tasks_tensor == task_name)
            
        if mask.sum() == 0: continue
        
        # 计算三个层级的指标
        id_r1, id_r5, id_r10 = calc_recall(gt_id_t, top10_indices, mask)
        st_r1, st_r5, st_r10 = calc_recall(gt_strict_t, top10_indices, mask)
        so_r1, so_r5, so_r10 = calc_recall(gt_soft_t, top10_indices, mask)
        
        results[task_name] = {
            "Count": mask.sum().item(),
            "ID":     {"R1": id_r1, "R5": id_r5},
            "Strict": {"R1": st_r1, "R5": st_r5, "R10": st_r10},
            "Soft":   {"R1": so_r1, "R5": so_r5}
        }
        
    return results

@torch.no_grad()
def evaluate():
    print(f"🚀 Loading Model: {CHECKPOINT}")
    model = GaitCIRModel().to(DEVICE)
    if os.path.exists(CHECKPOINT):
        model.combiner.load_state_dict(torch.load(CHECKPOINT))
    else:
        print("⚠️ 无权重，跑随机基线")
    model.eval()

    def collate_fn(batch):
        ref_imgs = torch.stack([item['ref_imgs'] for item in batch]) 
        tar_imgs = torch.stack([item['tar_imgs'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        masks = torch.stack([item['attention_mask'] for item in batch])
        tasks = [item['task'] for item in batch]
        # 收集元数据字典
        meta = [{'sid': item['sid'], 'cond': item['cond'], 'view': item['view']} for item in batch]
        return ref_imgs, tar_imgs, input_ids, masks, tasks, meta

    dataset = TestGaitDataset(TEST_JSON, DATA_ROOT, num_frames=NUM_FRAMES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    all_q, all_t, all_tasks, all_meta = [], [], [], []
    
    print("🔍 提取特征...")
    for ref, tar, ids, mask, tasks, meta in tqdm(loader):
        B, T, C, H, W = ref.shape
        ref = ref.view(B*T, C, H, W).to(DEVICE)
        tar = tar.view(B*T, C, H, W).to(DEVICE)
        ids = ids.to(DEVICE)
        mask = mask.to(DEVICE)
        
        with torch.cuda.amp.autocast():
            ref_f = model.extract_img_feature(ref).view(B, T, -1).mean(dim=1)
            tar_f = model.extract_img_feature(tar).view(B, T, -1).mean(dim=1)
            txt_f = model.extract_txt_feature(ids, mask)
            q_f = model.combiner(ref_f, txt_f)
        
        all_q.append(q_f.float().cpu())
        all_t.append(tar_f.float().cpu())
        all_tasks.extend(tasks)
        all_meta.extend(meta)
        
    all_q = torch.cat(all_q, dim=0)
    all_t = torch.cat(all_t, dim=0)
    
    # 计算分层指标
    metrics = compute_hierarchical_metrics(all_q, all_t, all_meta, all_meta, all_tasks)
    
    # 打印漂亮的报表
    print("\n" + "="*85)
    print(f"{'Task Type':<20} | {'Metric':<8} | {'R@1':<6} | {'R@5':<6} | {'R@10':<6}")
    print("-" * 85)
    
    order = ["attribute_change", "viewpoint_change", "composite_change", "Overall"]
    
    for task in order:
        if task in metrics:
            res = metrics[task]
            print(f"{task:<20} ({res['Count']})")
            
            # 打印 Strict (主指标)
            s = res['Strict']
            print(f"  {'':<20} | {'Strict':<8} | {s['R1']:.1f}   | {s['R5']:.1f}   | {s['R10']:.1f}")
            
            # 打印 Soft (辅助)
            so = res['Soft']
            print(f"  {'':<20} | {'Soft':<8} | {so['R1']:.1f}   | {so['R5']:.1f}   | -")
            
            # 打印 ID (基线)
            i = res['ID']
            print(f"  {'':<20} | {'ID-Only':<8} | {i['R1']:.1f}   | {i['R5']:.1f}   | -")
            print("-" * 40)
            
    print("="*85)

if __name__ == '__main__':
    evaluate()