import torch
import torch.nn.functional as F
import numpy as np

def compute_hierarchical_metrics(query_feats, gallery_feats, q_meta, g_meta, tasks):
    """
    计算分层指标 (Hierarchical Metrics)
    q_meta, g_meta: list of dict {'sid', 'cond', 'view'}
    """
    print("🚀 [Utils] 正在计算分层相似度矩阵...")
    
    # 1. 归一化 & 转 CPU
    query_feats = F.normalize(query_feats, dim=-1).cpu()
    gallery_feats = F.normalize(gallery_feats, dim=-1).cpu()
    
    # 2. 相似度与排序
    sim_matrix = query_feats @ gallery_feats.T
    _, indices = torch.sort(sim_matrix, dim=1, descending=True)
    
    # 3. 准备元数据向量
    q_sid = np.array([m['sid'] for m in q_meta])
    g_sid = np.array([m['sid'] for m in g_meta])
    q_cond = np.array([m['cond'] for m in q_meta])
    g_cond = np.array([m['cond'] for m in g_meta])
    q_view = np.array([m['view'] for m in q_meta])
    g_view = np.array([m['view'] for m in g_meta])
    
    # 4. 构建 Ground Truth 矩阵
    gt_id = (q_sid[:, None] == g_sid[None, :])
    # Strict: ID + Cond + View
    gt_strict = gt_id & (q_cond[:, None] == g_cond[None, :]) & (q_view[:, None] == g_view[None, :])
    # Soft: ID + Cond
    gt_soft = gt_id & (q_cond[:, None] == g_cond[None, :])
    
    gt_id_t = torch.from_numpy(gt_id)
    gt_strict_t = torch.from_numpy(gt_strict)
    gt_soft_t = torch.from_numpy(gt_soft)
    
    # 5. 内部计算函数
    def calc_recall(gt_matrix, topk_indices, mask):
        indices_masked = topk_indices[mask]
        gt_masked = gt_matrix[mask]
        if gt_masked.size(0) == 0: return 0.0, 0.0, 0.0
        
        hits = torch.gather(gt_masked, 1, indices_masked)
        r1 = hits[:, :1].any(dim=1).float().mean().item() * 100
        r5 = hits[:, :5].any(dim=1).float().mean().item() * 100
        r10 = hits[:, :10].any(dim=1).float().mean().item() * 100
        return r1, r5, r10

    # 6. 统计
    results = {}
    unique_tasks = set(tasks)
    unique_tasks.add("Overall")
    top10_indices = indices[:, :10]
    tasks_tensor = np.array(tasks)
    
    for task_name in unique_tasks:
        if task_name == "Overall":
            mask = torch.ones(len(tasks), dtype=torch.bool)
        else:
            mask = torch.from_numpy(tasks_tensor == task_name)
            
        if mask.sum() == 0: continue
        
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