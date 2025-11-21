import json
import random
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
META_FILE = './GaitCIR_RGB/meta_casiab_static.json'        # Step 02 的产出 (包含多样化的 view_text)
TEMPLATE_FILE = './GaitCIR_RGB/templates_instruction.json' # Step 03 的产出 (包含去代词的模板)
OUTPUT_TRAIN = './GaitCIR_RGB/casiab_cir_final_train.json'

MAX_PAIRS_PER_ID = 800  # 采样强度 (根据你的需求调整，500-1000 适合 CASIA-B)

# === 逻辑控制：粗粒度视角映射 ===
# 仅用于判断 "是否发生了视角变化"，防止生成微小角度变化的废话指令
COARSE_MAP = {
    "000": "front",
    "018": "front-side", "036": "front-side", "054": "front-side",
    "072": "side", "090": "side", "108": "side",
    "126": "back-side", "144": "back-side", "162": "back-side",
    "180": "back"
}
# ===========================================

def safe_fill_view(template, view_text):
    """
    安全填槽函数：只替换 {view}，保留 {subject}
    """
    # 移除 view_text 可能自带的 "view" 冗余 (可选，视 Step02 的字典而定)
    # 这里假设 Step02 生成的是 "side view", "profile view" 等完整短语，直接填入即可
    return template.replace("{view}", view_text)

def build():
    print("正在加载元数据和指令库...")
    with open(META_FILE, 'r') as f:
        meta_db = json.load(f)
    with open(TEMPLATE_FILE, 'r') as f:
        templates = json.load(f)
        
    # 1. 重建索引: group[sid][cond][view]
    data_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in meta_db:
        data_index[item['sid']][item['condition']][item['view']].append(item)
        
    all_triplets = []
    stats = defaultdict(int)
    
    sorted_ids = sorted(data_index.keys())
    print(f"🚀 开始组装三元组 (采样深度: {MAX_PAIRS_PER_ID})...")
    
    for sid in tqdm(sorted_ids):
        conds = data_index[sid]
        
        # 收集所有可用节点 (Condition, View)
        nodes = []
        for c in conds:
            for v in conds[c]:
                if len(conds[c][v]) > 0:
                    nodes.append((c, v))
        
        if len(nodes) < 2: continue

        # --- 随机配对采样 ---
        for _ in range(MAX_PAIRS_PER_ID):
            src_node = random.choice(nodes)
            tgt_node = random.choice(nodes)
            
            if src_node == tgt_node: continue
            
            src_c, src_v = src_node
            tgt_c, tgt_v = tgt_node
            
            # 随机选具体序列
            ref_item = random.choice(conds[src_c][src_v])
            tar_item = random.choice(conds[tgt_c][tgt_v])
            
            if ref_item['seq_path'] == tar_item['seq_path']: continue

            # === 核心逻辑判定 ===
            
            # 1. 判断状态指令 (State Instruction)
            state_instr = ""
            # NM <-> BG
            if src_c == 'nm' and tgt_c == 'bg':
                state_instr = random.choice(templates['source_nm_target_bg'])
            elif src_c == 'bg' and tgt_c == 'nm':
                state_instr = random.choice(templates['source_bg_target_nm'])
            # NM <-> CL
            elif src_c == 'nm' and tgt_c == 'cl':
                state_instr = random.choice(templates['source_nm_target_cl'])
            elif src_c == 'cl' and tgt_c == 'nm':
                state_instr = random.choice(templates['source_cl_target_nm'])
            # BG <-> CL (互换)
            elif src_c == 'bg' and tgt_c == 'cl':
                state_instr = random.choice(templates['source_bg_target_cl'])
            elif src_c == 'cl' and tgt_c == 'bg':
                state_instr = random.choice(templates['source_cl_target_bg'])
            
            # 2. 判断视角指令 (View Instruction)
            view_instr = ""
            
            # 【关键】使用粗粒度逻辑判断是否发生了有意义的视角变化
            src_coarse = COARSE_MAP.get(src_v, src_v)
            tgt_coarse = COARSE_MAP.get(tgt_v, tgt_v)
            
            if src_coarse != tgt_coarse:
                # 发生了大的视角变化 -> 生成指令
                tpl = random.choice(templates['change_view'])
                # 【关键】填槽使用 Step02 带来的多样化描述 (e.g., "profile view")
                view_instr = safe_fill_view(tpl, tar_item['view_text'])

            # 3. 组装最终 Caption
            final_caption = ""
            task_type = "unknown"
            
            # Case A: 复合变换 (Composite)
            if state_instr and view_instr:
                conn = random.choice(templates['connectors'])
                # 去标点 + 小写化拼接
                s_part = state_instr.rstrip('.')
                v_part = view_instr[0].lower() + view_instr[1:] 
                if v_part.endswith('.'): v_part = v_part[:-1]
                
                final_caption = f"{s_part}{conn}{v_part}."
                task_type = "composite_change"
                
            # Case B: 仅属性变换 (Attribute)
            elif state_instr:
                # 如果视角没大变 (coarse same)，我们就认为只是换属性
                final_caption = state_instr
                task_type = "attribute_change"
                
            # Case C: 仅视角变换 (Viewpoint)
            elif view_instr and src_c == tgt_c:
                final_caption = view_instr
                task_type = "viewpoint_change"
            
            else:
                # 1. 状态没变 + 视角属于同一粗粒度 (例如 036 -> 054) -> 跳过
                # 2. 状态变了但没模板 (异常) -> 跳过
                continue

            # 4. 添加到数据集
            all_triplets.append({
                "sid": sid,
                "dataset": "CASIA-B",
                "task": task_type,
                "caption": final_caption, # 这里的 caption 包含 {subject}，等待训练时替换
                "ref": ref_item,          # 包含多样化的 static_caption
                "tar": tar_item
            })
            stats[task_type] += 1

    # 保存结果
    with open(OUTPUT_TRAIN, 'w') as f:
        json.dump(all_triplets, f, indent=4)
    
    print(f"✅ 生成完毕! 总样本量: {len(all_triplets)}")
    print("📊 任务分布:", dict(stats))

if __name__ == '__main__':
    build()