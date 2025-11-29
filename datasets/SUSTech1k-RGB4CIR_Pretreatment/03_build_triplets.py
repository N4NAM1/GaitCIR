import json
import random
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

# ================= 辅助函数区域 =================

def parse_status_attributes(status_str):
    """
    解析 CCPG 状态属性 (粗粒度逻辑)
    Args:
        status_str: 文件夹名称, 例如 'U0_D0', 'U1_D0_BG'
    Returns:
        clothing_id (str): 服装标识 (去除BG后的字符串, 用作唯一服装ID)
        has_bag (bool): 是否有包
    """
    # 【修复】: 增加对 None 的防御性检查，防止 crash
    if status_str is None:
        return "Unknown", False

    has_bag = "BG" in status_str
    # 粗粒度逻辑：只要去除BG后缀后的字符串不同，就视为服装不同
    # 例如: U0_D0 vs U1_D0 是不同的衣服
    clothing_id = status_str.replace("_BG", "").replace("BG", "")
    return clothing_id, has_bag

def get_template_text(templates, key):
    """从模板库中安全随机获取一条指令文本"""
    if key in templates and len(templates[key]) > 0:
        return random.choice(templates[key])
    return ""

def clean_text(text):
    """
    【新增】清理文本末尾的标点符号，方便拼接
    """
    if not text: return ""
    return text.strip().rstrip('.!?')

def combine_texts(texts, templates):
    """
    使用连接词拼接多段文本，用于构建混合或组合指令
    """
    # 【修改】: 拼接前先清理掉每个子句末尾的标点
    valid_texts = [clean_text(t) for t in texts if t]
    
    if not valid_texts: return ""
    
    # 随机打乱语序 (例如: 先说视角还是先说换衣，增加数据多样性)
    random.shuffle(valid_texts)
    
    if len(valid_texts) == 1:
        combined = valid_texts[0]
    else:
        # 【修改】: 支持多段文本使用不同的随机连接词
        combined = valid_texts[0]
        connectors_pool = templates.get("connectors", [" and "])
        
        for i in range(1, len(valid_texts)):
            # 每次循环都重新随机选一个连接词
            connector = random.choice(connectors_pool)
            combined += connector + valid_texts[i]
    
    # 格式简单修正
    combined = combined.strip()
    
    # 确保首字母大写 (如果开头是占位符 {subject} 则不处理，否则大写)
    if len(combined) > 0 and not combined.startswith("{"):
        combined = combined[0].upper() + combined[1:]
    
    # 确保结尾有标点
    if len(combined) > 0 and combined[-1] not in ['.', '!', '?', '}']:
        combined += '.'
        
    return combined

def create_static_caption(pid, status, view):
    """
    生成静态描述的占位符 (Ref/Tar 内部使用)
    注意：这里保留了 {subject} 占位符
    """
    clothing_id, has_bag = parse_status_attributes(status)
    
    # 构造描述部件
    parts = []
    
    # 1. 衣服描述 (粗粒度规则)
    if clothing_id == "U0_D0":
        parts.append("in standard outfit")
    else:
        parts.append("wearing different clothes")
        
    # 2. 包描述
    if has_bag:
        parts.append("carrying a bag")
    else:
        parts.append("without a bag")
        
    # 3. 视角描述 (Blind View 使用 Camera ID 作为占位)
    view_desc = f"viewed from camera {view}"
    
    # 组合: "{subject}, in standard outfit, without a bag, viewed from camera 01."
    desc = "{subject}, " + ", ".join(parts) + ", " + view_desc + "."
    return desc

def generate_instruction_pair(ref_item, tar_item, templates):
    """
    核心指令生成逻辑：判断 Ref 和 Tar 之间的差异，生成正向(fwd)和反向(inv)指令
    
    Returns: 
        final_caption (str): 正向指令
        final_caption_inv (str): 反向指令
        task_type (str): 任务类型 (viewpoint_change / attribute_change / composite_change)
    """
    # 1. 解析属性差异
    ref_cloth, ref_bag = ref_item['parsed_attr']
    tar_cloth, tar_bag = tar_item['parsed_attr']
    
    # --- A. 状态部分 (Attribute Instruction) ---
    attr_tasks_fwd = []
    attr_tasks_inv = []
    
    # A1. 换衣判定 (只要 Clothing ID 不同即为换衣)
    if ref_cloth != tar_cloth:
        attr_tasks_fwd.append("change_cloth")
        attr_tasks_inv.append("change_cloth") # 粗粒度下换衣是对称操作
        
    # A2. 换包判定
    if ref_bag != tar_bag:
        if not ref_bag and tar_bag: # 加包 (Add Bag)
            attr_tasks_fwd.append("change_bag_add")
            attr_tasks_inv.append("change_bag_remove")
        else: # 去包 (Remove Bag)
            attr_tasks_fwd.append("change_bag_remove")
            attr_tasks_inv.append("change_bag_add")
            
    # 生成状态文本列表 (List of strings)
    state_parts_fwd = [get_template_text(templates, t) for t in attr_tasks_fwd]
    state_parts_inv = [get_template_text(templates, t) for t in attr_tasks_inv]

    # --- B. 视角部分 (View Instruction) ---
    view_parts_fwd = []
    view_parts_inv = []
    
    # 视角判定 (Blind View: 只要 view code 不同即为变视角)
    has_view_change = ref_item.get('view') != tar_item.get('view')
    if has_view_change:
        v_txt = get_template_text(templates, "change_view")
        view_parts_fwd.append(v_txt)
        view_parts_inv.append(v_txt) # 视角变化是对称操作

    # --- C. 任务类型判定与最终文本组装 ---
    final_caption = ""
    final_caption_inv = ""
    task_type = "unknown"

    has_state_change = len(state_parts_fwd) > 0

    # 逻辑核心：只有两者都变才算 composite
    if has_state_change and has_view_change:
        task_type = "composite_change"
        # 拼接所有部分 (属性 + 视角)
        final_caption = combine_texts(state_parts_fwd + view_parts_fwd, templates)
        final_caption_inv = combine_texts(state_parts_inv + view_parts_inv, templates)
        
    elif has_state_change:
        # 只有属性变了 (可能是单纯换衣，单纯换包，或者混合换衣换包)
        task_type = "attribute_change"
        final_caption = combine_texts(state_parts_fwd, templates)
        final_caption_inv = combine_texts(state_parts_inv, templates)
        
    elif has_view_change:
        # 只有视角变了
        task_type = "viewpoint_change"
        final_caption = combine_texts(view_parts_fwd, templates)
        final_caption_inv = combine_texts(view_parts_inv, templates)
        
    else:
        # 既没变状态也没变视角 (跳过)
        return None, None, None

    return final_caption, final_caption_inv, task_type

def create_entry(sid, ref_meta, tar_meta, task_type, caption, caption_inv):
    """
    构建符合 CASIA-B CIR 格式的 JSON 条目 (嵌套结构)
    """
    # 【修复】: 先解析状态，兼容 'condition' 或 'status' 键名
    ref_status = ref_meta.get('condition', ref_meta.get('status'))
    tar_status = tar_meta.get('condition', tar_meta.get('status'))
    
    # 【修复】: 获取静态描述时，使用已经解析好的 ref_status，而不是再去 get('status')
    ref_static = ref_meta.get('static_caption')
    if not ref_static:
        ref_static = create_static_caption(sid, ref_status, ref_meta.get('view'))
        
    tar_static = tar_meta.get('static_caption')
    if not tar_static:
        tar_static = create_static_caption(sid, tar_status, tar_meta.get('view'))

    return {
        "sid": sid,
        "dataset": "CCPG",
        "task": task_type,
        "caption": caption,         # 正向指令 (保留 {subject})
        "caption_inv": caption_inv, # 反向指令 (保留 {subject})
        "ref": {
            "sid": sid,
            "condition": ref_status, # 使用解析好的值
            "view": ref_meta.get('view'),
            "seq_path": ref_meta.get('seq_path', ref_meta.get('file_path')),
            "static_caption": ref_static
        },
        "tar": {
            "sid": sid,
            "condition": tar_status, # 使用解析好的值
            "view": tar_meta.get('view'),
            "seq_path": tar_meta.get('seq_path', tar_meta.get('file_path')),
            "static_caption": tar_static
        }
    }

# ================= 主逻辑区域 =================

def build_ccpg_triplets(args):
    # 1. 加载元数据和模板
    print(f"正在加载元数据: {args.meta_path} ...")
    with open(args.meta_path, 'r') as f:
        meta_data = json.load(f)
    print(f"正在加载指令模板: {args.template_path} ...")
    with open(args.template_path, 'r') as f:
        templates = json.load(f)

    # 2. 按 Subject ID 分组数据 (重建索引)
    pid_groups = defaultdict(list)
    for key, info in meta_data.items():
        # 兼容 meta 文件中的 sid 或 pid 字段
        sid = info.get('sid', info.get('pid'))
        
        # 预先解析属性，避免循环中重复计算
        # 优先取 'condition'，如果不存在取 'status'，再不存在给 None
        status = info.get('condition', info.get('status'))
        info['parsed_attr'] = parse_status_attributes(status)
        
        pid_groups[sid].append(info)

    # 3. 划分训练集和测试集 ID
    # 按照 CCPG 惯例，前 N 个 ID 通常用于训练
    all_sids = sorted(list(pid_groups.keys()))
    train_sids = set(all_sids[:args.train_ids_count])
    
    print(f"总 ID 数: {len(all_sids)}. 训练集 ID 数: {len(train_sids)} (采样{args.sample_train}/ID), 测试集 ID 数: {len(all_sids)-len(train_sids)} (采样{args.sample_test}/ID).")
    
    all_triplets = []
    stats = defaultdict(int)

    # 4. 遍历每个 Subject 生成 Pair
    for sid in tqdm(all_sids, desc="生成三元组"):
        sequences = pid_groups[sid]
        # 如果一个 ID 下序列少于2个，无法构建 Pair，跳过
        if len(sequences) < 2: continue

        # 确定当前 ID 的采样上限
        is_train = sid in train_sids
        sample_limit = args.sample_train if is_train else args.sample_test
        
        generated_count = 0
        attempts = 0
        # 防止死循环的安全阈值 (比如该 ID 下只有2个样本，不可能生成400对)
        max_attempts = sample_limit * 10 

        while generated_count < sample_limit and attempts < max_attempts:
            attempts += 1
            
            # A. 随机选择 Reference 和 Target
            ref = random.choice(sequences)
            tar = random.choice(sequences)
            
            # 排除自引用 (Source 和 Target 是同一个序列)
            if ref['seq_path'] == tar['seq_path']: continue
            
            # B. 生成指令与任务判定 (核心逻辑)
            caption, caption_inv, task_type = generate_instruction_pair(ref, tar, templates)
            
            # 如果 caption 为 None，说明既没变属性也没变视角 (例如同状态同视角的不同序列)，跳过
            if not caption: continue 

            # C. 保存结果
            entry = create_entry(sid, ref, tar, task_type, caption, caption_inv)
            all_triplets.append(entry)
            
            stats[task_type] += 1
            generated_count += 1

    # 5. 保存结果到单一文件
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_triplets, f, indent=4)
    
    print(f"✅ 生成完毕! 总样本量: {len(all_triplets)}")
    print(f"💾 已保存至: {args.output}")
    print("📊 任务类型分布:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CCPG Triplets for GaitCIR Task")
    
    # 输入路径
    parser.add_argument('--meta_path', default='datasets/CCPG/meta_ccpg.json', help='Step 02 生成的元数据路径')
    parser.add_argument('--template_path', default='datasets/CCPG/templates_instruction.json', help='指令模板路径')
    
    # 输出路径
    parser.add_argument('--output', default='datasets/CCPG/ccpg_cir_final.json', help='最终生成的训练/测试整合 JSON')
    
    # 采样与分割配置
    parser.add_argument('--train_ids_count', type=int, default=100, help='前 N 个 ID 划分为训练集 (CCPG 默认 100)')
    parser.add_argument('--sample_train', type=int, default=400, help='训练集每个 ID 采样的 Pair 数量')
    parser.add_argument('--sample_test', type=int, default=100, help='测试集每个 ID 采样的 Pair 数量')
    
    args = parser.parse_args()
    
    # 执行主程序
    build_ccpg_triplets(args)