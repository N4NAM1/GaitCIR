import torch
import numpy as np
from collections import defaultdict

def compute_mAP_vectorized(sim_mat, is_match):
    """
    [GPU 加速] 计算 Mean Average Precision (mAP)
    适用于单 GT 和多 GT (Soft Match) 的情况。
    
    Args:
        sim_mat: [N_query, N_gallery] 相似度矩阵 (越大越好)
        is_match: [N_query, N_gallery] Boolean/Int (Ground Truth 掩码)
    """
    # 1. 排序: 相似度从高到低 (descending=True)
    scores, indices = torch.sort(sim_mat, dim=1, descending=True)
    
    # 2. 根据排序结果重排 Ground Truth
    # gts[i][j] = 1 表示第 i 个 Query 的第 j 名预测结果是正确的
    gts = torch.gather(is_match.float(), 1, indices)
    
    # 3. 计算 Precision
    # cumsum: 计算每一位的累计正确数 (即 TP @ k)
    cumsum = torch.cumsum(gts, dim=1)
    
    # ranks: 生成排名 [1, 2, 3, ..., M]
    ranks = torch.arange(1, gts.size(1) + 1, device=gts.device).float().unsqueeze(0)
    
    # precision @ k = (TP @ k) / k
    precision = cumsum / ranks
    
    # 4. 计算 Average Precision (AP)
    # AP = sum(precision * is_relevant) / num_total_relevant
    # 只有相关位置的 precision 才参与求和
    ap_sum = (precision * gts).sum(dim=1)
    num_relevant = gts.sum(dim=1)
    
    # 5. 避免除以 0 (对于没有任何正确答案的 Query，AP=0)
    mask = num_relevant > 0
    ap = torch.zeros_like(ap_sum)
    ap[mask] = ap_sum[mask] / num_relevant[mask]
    
    # 6. Mean AP
    return ap.mean().item()

def compute_rank_k_vectorized(sim_mat, is_match, ks=[1, 5, 10]):
    """
    [GPU 加速] 计算 R@K (CMC Rank-k / Hit Rate)
    定义：只要 Top-K 结果中包含 *至少一个* 正确答案，就算命中 (Hit)。
    """
    # 1. 排序
    _, indices = torch.sort(sim_mat, dim=1, descending=True)
    
    # 2. 重排 GT
    gts = torch.gather(is_match.float(), 1, indices)
    
    results = {}
    for k in ks:
        if k > gts.size(1):
            results[f'R{k}'] = 100.0
            continue
            
        # 取前 K 列
        top_k = gts[:, :k]
        
        # 只要行和 > 0，说明命中了至少 1 个
        hits = (top_k.sum(dim=1) > 0).float()
        
        # 平均命中率
        results[f'R{k}'] = hits.mean().item() * 100.0
        
    return results

def _compute_on_device(device, q_feats, g_feats, q_meta, g_meta, q_tasks):
    """
    内部计算函数：负责数据搬运、Mask 生成和指标计算
    """
    # 1. 特征搬运与归一化
    q_feats = q_feats.to(device)
    g_feats = g_feats.to(device)
    
    q_feats = torch.nn.functional.normalize(q_feats, p=2, dim=1)
    g_feats = torch.nn.functional.normalize(g_feats, p=2, dim=1)
    
    # 2. 计算相似度矩阵
    sim_mat = torch.mm(q_feats, g_feats.t())
    
    # 3. 提取元数据 (CPU Numpy 处理字符串，生成 Bool Mask 后转 GPU)
    q_sids = np.array([m['sid'] for m in q_meta])
    g_sids = np.array([m['sid'] for m in g_meta])
    
    q_views = np.array([m['view'] for m in q_meta])
    g_views = np.array([m['view'] for m in g_meta])
    
    # === 关键操作：模糊匹配 Condition (忽略序列号，例如 'bg-01' -> 'bg') ===
    q_conds = np.array([str(m['cond']).split('-')[0] for m in q_meta])
    g_conds = np.array([str(m['cond']).split('-')[0] for m in g_meta])
    
    # 4. 生成 GPU 布尔掩码 (Ground Truth)
    # Broadcasting: [N, 1] == [1, M] -> [N, M]
    match_id = torch.from_numpy(q_sids[:, None] == g_sids[None, :]).to(device)
    match_cond = torch.from_numpy(q_conds[:, None] == g_conds[None, :]).to(device)
    match_view = torch.from_numpy(q_views[:, None] == g_views[None, :]).to(device)
    
    # 5. 分任务计算
    task_groups = defaultdict(list)
    for idx, task in enumerate(q_tasks):
        task_groups[task].append(idx)
        
    # 增加 Overall 组
    if "Overall" not in task_groups:
        task_groups["Overall"] = list(range(len(q_feats)))
        
    final_results = {}
    
    for task, indices in task_groups.items():
        if len(indices) == 0: continue
        
        # 提取当前任务的子集索引
        indices_t = torch.tensor(indices, device=device)
        
        # 切片矩阵
        sub_sim = sim_mat[indices_t]
        sub_match_id = match_id[indices_t]
        sub_match_cond = match_cond[indices_t]
        sub_match_view = match_view[indices_t]
        
        # === 定义匹配标准 (Soft 匹配根据任务动态调整) ===
        
        # A. Strict: 严格匹配 (ID & 模糊化 Condition & View)
        # 你的要求：ID 对 + 语义状态对 + 视角对，忽略序列号。
        gt_strict = sub_match_id & sub_match_cond & sub_match_view
        
        # B. Soft: 宽松匹配 (根据任务聚焦指令要求)
        if task == 'viewpoint_change':
            # 视角任务：只需要 ID 对 + 视角对 (忽略属性差异)
            gt_soft = sub_match_id & sub_match_view
        elif task == 'attribute_change':
            # 属性任务：只需要 ID 对 + 属性/状态对 (忽略视角差异)
            gt_soft = sub_match_id & sub_match_cond
        elif task == 'composite_change':
            # 复合任务：需要 ID 对 + 属性/状态对 + 视角对
            # Soft在于它仅忽略序列号。
            gt_soft = sub_match_id & sub_match_cond & sub_match_view
        else:
            # Overall 或其他任务：使用最通用的 Soft 定义 (ID + 属性)
            gt_soft = sub_match_id & sub_match_cond
            
        # C. ID: 身份匹配 (人对)
        gt_id = sub_match_id
        
        # === 计算指标 ===
        metrics = {'Count': len(indices)}
        
        # 1. Strict
        res_strict = compute_rank_k_vectorized(sub_sim, gt_strict)
        res_strict['mAP'] = compute_mAP_vectorized(sub_sim, gt_strict) * 100
        metrics['Strict'] = res_strict
        
        # 2. Soft
        res_soft = compute_rank_k_vectorized(sub_sim, gt_soft)
        res_soft['mAP'] = compute_mAP_vectorized(sub_sim, gt_soft) * 100
        metrics['Soft'] = res_soft
        
        # 3. ID
        res_id = compute_rank_k_vectorized(sub_sim, gt_id)
        res_id['mAP'] = compute_mAP_vectorized(sub_sim, gt_id) * 100
        metrics['ID'] = res_id
        
        final_results[task] = metrics
        
    return final_results

def compute_hierarchical_metrics(q_feats, g_feats, q_meta, g_meta, q_tasks):
    """
    对外接口：自动显存保护
    """
    try:
        # 尝试使用 CUDA
        if torch.cuda.is_available():
            return _compute_on_device(torch.device("cuda"), q_feats, g_feats, q_meta, g_meta, q_tasks)
        else:
            # 使用 CPU
            return _compute_on_device(torch.device("cpu"), q_feats, g_feats, q_meta, g_meta, q_tasks)
    except RuntimeError as e:
        # 捕捉 CUDA 内存不足错误，切换到 CPU
        if "out of memory" in str(e):
            print("\n⚠️ [Metrics] GPU OOM! Swapping to CPU for evaluation...")
            torch.cuda.empty_cache()
            return _compute_on_device(torch.device("cpu"), q_feats, g_feats, q_meta, g_meta, q_tasks)
        else:
            raise e