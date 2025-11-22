import json
import random
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
META_FILE = './GaitCIR_RGB/meta_casiab_static.json'        
TEMPLATE_FILE = './GaitCIR_RGB/templates_instruction.json' 
OUTPUT_TRAIN = './GaitCIR_RGB/casiab_cir_final.json'

MAX_PAIRS_PER_ID = 800  # 采样强度

# === 1. 视角描述池 (动态增强用) ===
# 替代了原先存在 item['view_text'] 里的固定文本

VIEW_TEXT_POOL = {
    "000": [
        "a front view", "a frontal angle", "a 0-degree view", "a face-to-face view",
        "a view facing the camera", 
        "a head-on view", "a straight-on shot", "a view facing forward"
    ],
    "018": [
        "a front-side view", "an oblique angle", 
        "a slight frontal angle", "a front-quarter view", 
        "a view walking towards at an angle", "an angled front view"
    ],
    "036": [
        "a front-side view", "an oblique angle",
        "a front-quarter view", "a diagonal front view", "an angled view from the front"
    ],
    "054": [
        "a front-side view", "an oblique angle",
        "a sharp frontal angle", "a semi-frontal view", "an off-center front view"
    ],
    "072": [
        "a side view", "a profile view",
        "a slight profile", "a near-side view", "a view turning to the side", 
        "a view almost from the side" 
    ],
    "090": [
        "a side view", "a profile view", "a lateral view", "a 90-degree view",
        "a side-on view", "a view from the side", "a full profile", 
        "a shot walking sideways"
    ],
    "108": [
        "a side view", "a profile view", "a 108-degree view",
        "a past-side view", "an angled profile"
    ],
    "126": [
        "a back-side view", "a rear-oblique view",
        "a rear-quarter view", "a view walking away at an angle", 
        "a diagonal back view", "an angled rear view"
    ],
    "144": [
        "a back-side view", "a rear-oblique view",
        "a rear-quarter view", "an off-center back view", "a view from behind and side"
    ],
    "162": [
        "a back-side view", "a rear-oblique view",
        "a slight rear angle", "an almost back view", "a view turning away"
    ],
    "180": [
        "a back view", "a rear view", "a dorsal view", "a 180-degree view",
        "a view seen from behind", "a shot walking away", "a view with the back turned", 
        "a view of the back", "a straight back view"
    ]
}

# === 2. 粗粒度映射 (仅用于逻辑判断) ===
COARSE_MAP = {
    "000": "front",
    "018": "front-side", "036": "front-side", "054": "front-side",
    "072": "side", "090": "side", "108": "side",
    "126": "back-side", "144": "back-side", "162": "back-side",
    "180": "back"
}

def safe_fill_view(template, view_text):
    """
    安全填槽：将模板中的 {view} 替换为具体的 view_text
    """
    return template.replace("{view}", view_text)

def get_instruction(src_item, tgt_item, templates):
    """
    核心指令生成函数
    Args:
        src_item: 源样本信息
        tgt_item: 目标样本信息
        templates: 模板字典
    Returns:
        final_caption: 生成的文本指令
        task_type: 任务类型
    """
    src_c, src_v = src_item['condition'], src_item['view']
    tgt_c, tgt_v = tgt_item['condition'], tgt_item['view']

    # --- A. 状态部分 (State Instruction) ---
    state_instr = ""
    
    # 根据 Condition 组合选择对应模板
    # 注意：这里假设 templates 的键名与你的 templates_instruction.json 一致
    if src_c == 'nm' and tgt_c == 'bg':
        state_instr = random.choice(templates['source_nm_target_bg'])
    elif src_c == 'bg' and tgt_c == 'nm':
        state_instr = random.choice(templates['source_bg_target_nm'])
    elif src_c == 'nm' and tgt_c == 'cl':
        state_instr = random.choice(templates['source_nm_target_cl'])
    elif src_c == 'cl' and tgt_c == 'nm':
        state_instr = random.choice(templates['source_cl_target_nm'])
    elif src_c == 'bg' and tgt_c == 'cl':
        state_instr = random.choice(templates['source_bg_target_cl'])
    elif src_c == 'cl' and tgt_c == 'bg':
        state_instr = random.choice(templates['source_cl_target_bg'])
    
    # --- B. 视角部分 (View Instruction) ---
    view_instr = ""
    
    src_coarse = COARSE_MAP.get(src_v, src_v)
    tgt_coarse = COARSE_MAP.get(tgt_v, tgt_v)
    
    # 只有当粗粒度视角发生变化时，才生成视角指令
    if src_coarse != tgt_coarse:
        tpl = random.choice(templates['change_view'])
        
        # 【修复点】: 动态从池中获取描述，不再依赖 item['view_text']
        tgt_angle = tgt_item['view'] # e.g., "090"
        # 使用 .get 默认值防止 KeyError
        potential_texts = VIEW_TEXT_POOL.get(tgt_angle, [f"{tgt_angle} degree view"]) 
        view_text = random.choice(potential_texts)
        
        view_instr = safe_fill_view(tpl, view_text)

    # --- C. 组装最终指令 ---
    final_caption = ""
    task_type = "unknown"
    
    # Case 1: 复合变换 (Composite: State + View)
    if state_instr and view_instr:
        conn = random.choice(templates['connectors'])
        
        # 移除末尾可能存在的标点，方便拼接
        s_text = state_instr.rstrip('.')
        v_text = view_instr.rstrip('.') 
        
        # 【随机语序】防止模型过拟合特定句式
        if random.random() > 0.5:
            # 顺序: State + Conn + View (e.g., "Wear a coat and turn to side")
            p1 = s_text
            # 处理 p2 首字母小写 (如果 p2 是 "Side view" -> "side view")
            p2_content = v_text[0].lower() + v_text[1:] if len(v_text) > 0 else ""
            p2 = p2_content
        else:
            # 顺序: View + Conn + State (e.g., "Turn to side and wear a coat")
            p1 = v_text
            p2_content = s_text[0].lower() + s_text[1:] if len(s_text) > 0 else ""
            p2 = p2_content
            
        final_caption = f"{p1}{conn}{p2}."
        task_type = "composite_change"
        
    # Case 2: 仅属性变换 (Attribute Only)
    elif state_instr:
        final_caption = state_instr
        task_type = "attribute_change"
        
    # Case 3: 仅视角变换 (Viewpoint Only)
    elif view_instr and src_c == tgt_c:
        final_caption = view_instr
        task_type = "viewpoint_change"
    
    # Case 4: 既没变属性也没变视角 (通常在循环中会被 continue 跳过)
    
    return final_caption, task_type

def build():
    print("正在加载元数据和指令库...")
    try:
        with open(META_FILE, 'r') as f:
            meta_db = json.load(f)
        with open(TEMPLATE_FILE, 'r') as f:
            templates = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件 {e.filename}。请确保 02 步已运行且路径正确。")
        return

    # 1. 重建索引: group[sid][cond][view] -> list of items
    data_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in meta_db:
        data_index[item['sid']][item['condition']][item['view']].append(item)
        
    all_triplets = []
    stats = defaultdict(int)
    
    sorted_ids = sorted(data_index.keys())
    print(f"🚀 开始组装三元组 (采样深度: {MAX_PAIRS_PER_ID})...")
    
    for sid in tqdm(sorted_ids):
        conds = data_index[sid]
        
        # 收集该 ID 下所有可用的 (Condition, View) 节点
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
            
            # 从具体序列中随机选择图片 (例如 nm-01 和 nm-02)
            ref_item = random.choice(conds[src_c][src_v])
            tar_item = random.choice(conds[tgt_c][tgt_v])
            
            # 避免同一序列自检索 (Optional)
            if ref_item['seq_path'] == tar_item['seq_path']: continue

            # === 核心逻辑 ===
            
            # 1. 生成正向指令 (Ref -> Tar)
            fwd_caption, task_type = get_instruction(ref_item, tar_item, templates)
            
            # 如果没生成出指令 (例如视角没大变且状态也没变)，则跳过该对
            if not fwd_caption: continue

            # 2. 生成逆向指令 (Tar -> Ref) 【新增用于 Cycle Loss】
            # 直接复用 get_instruction，只是源和目标互换
            inv_caption, _ = get_instruction(tar_item, ref_item, templates)

            # 3. 保存
            all_triplets.append({
                "sid": sid,
                "dataset": "CASIA-B",
                "task": task_type,
                "caption": fwd_caption,       # 正向指令 (Input)
                "caption_inv": inv_caption,   # 逆向指令 (Cycle Constraint)
                "ref": ref_item,              # Ref Image Info
                "tar": tar_item               # Target Image Info
            })
            stats[task_type] += 1

    # 保存结果
    with open(OUTPUT_TRAIN, 'w') as f:
        json.dump(all_triplets, f, indent=4)
    
    print(f"✅ 生成完毕! 总样本量: {len(all_triplets)}")
    print("📊 任务分布:", dict(stats))

if __name__ == '__main__':
    build()