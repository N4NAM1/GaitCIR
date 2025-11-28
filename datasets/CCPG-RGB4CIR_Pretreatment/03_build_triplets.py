import json
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
import os

def parse_status_attributes(status_str):
    """
    解析 CCPG 状态属性
    返回: (clothing_id, has_bag)
    """
    has_bag = "BG" in status_str
    # 提取服装ID (去除 BG 后剩余部分即为服装标识, 如 U0_D0_BG -> U0_D0)
    clothing_id = status_str.replace("_BG", "").replace("BG", "")
    return clothing_id, has_bag

def get_instruction(templates, task_key):
    """从模板中随机获取一条指令"""
    if task_key in templates:
        return random.choice(templates[task_key])
    return ""

def build_composite_instruction(templates, tasks):
    """
    动态拼接多个子任务指令
    tasks: list of task keys, e.g. ['change_cloth', 'change_bag_add']
    """
    if not tasks: return ""
    
    # 获取单条指令
    instr_list = [get_instruction(templates, t) for t in tasks if t in templates]
    
    if not instr_list: return ""
    
    # 随机打乱顺序 (先说换衣还是先说换包)
    random.shuffle(instr_list)
    
    # 随机选择连接词
    connector = random.choice(templates.get("connectors", [" and "]))
    
    # 拼接
    combined = connector.join(instr_list)
    
    # 简单的格式修正
    combined = combined.strip()
    # 确保首字母大写
    combined = combined[0].upper() + combined[1:]
    # 确保结尾有标点
    if combined[-1] not in ['.', '!', '?']:
        combined += '.'
        
    return combined

def build_ccpg_cir_json(meta_path, template_path, output_path):
    # 1. 加载元数据和模板
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    with open(template_path, 'r') as f:
        templates = json.load(f)

    # 2. 按 Subject ID 分组
    pid_groups = defaultdict(list)
    for key, info in meta_data.items():
        # 兼容 key 'sid' 或 'pid'
        sid = info.get('sid', info.get('pid'))
        pid_groups[sid].append(info)

    triplets = []
    stats = defaultdict(int)

    # 3. 遍历每个 Subject 生成 Pair
    for sid, sequences in tqdm(pid_groups.items(), desc="Generating CCPG Triplets"):
        
        # 预解析属性
        for seq in sequences:
            # 兼容 key 'condition' 或 'status'
            status = seq.get('condition', seq.get('status'))
            seq['parsed_attr'] = parse_status_attributes(status)

        # 遍历每个序列作为 Reference (Anchor)
        for ref in sequences:
            ref_cloth, ref_bag = ref['parsed_attr']
            ref_view = ref.get('view')
            
            # --- 寻找合适的 Target ---
            # 为了控制数据规模，我们在每个类别中随机采样 1-2 个 target
            
            # === 1. View Change (Task: view_change) ===
            # 条件: 同衣、同包、异视角
            view_candidates = [s for s in sequences if s['parsed_attr'] == (ref_cloth, ref_bag) and s.get('view') != ref_view]
            if view_candidates:
                tar = random.choice(view_candidates)
                
                # 正向指令
                caption = get_instruction(templates, "change_view")
                # 反向指令 (视角变化是对称的，也是 change_view)
                caption_inv = get_instruction(templates, "change_view")
                
                triplets.append(create_entry(sid, ref, tar, "view_change", caption, caption_inv))
                stats['view_change'] += 1

            # === 2. Attribute Change (Task: attribute_change) ===
            
            # 2.1 仅换衣 (Cloth Change)
            # 条件: 异衣、同包
            cloth_candidates = [s for s in sequences if s['parsed_attr'][0] != ref_cloth and s['parsed_attr'][1] == ref_bag]
            if cloth_candidates:
                tar = random.choice(cloth_candidates)
                
                # 正向: Change Cloth
                caption = get_instruction(templates, "change_cloth")
                # 反向: Change Cloth (换衣是对称操作)
                caption_inv = get_instruction(templates, "change_cloth")
                
                triplets.append(create_entry(sid, ref, tar, "attribute_change", caption, caption_inv))
                stats['attr_cloth'] += 1
            
            # 2.2 仅换包 (Bag Change)
            # 条件: 同衣、异包
            bag_candidates = [s for s in sequences if s['parsed_attr'][0] == ref_cloth and s['parsed_attr'][1] != ref_bag]
            if bag_candidates:
                tar = random.choice(bag_candidates)
                
                # 判断正向是加包还是去包
                is_add = (not ref_bag) and tar['parsed_attr'][1]
                
                fwd_task = "change_bag_add" if is_add else "change_bag_remove"
                rev_task = "change_bag_remove" if is_add else "change_bag_add"
                
                caption = get_instruction(templates, fwd_task)
                caption_inv = get_instruction(templates, rev_task)
                
                triplets.append(create_entry(sid, ref, tar, "attribute_change", caption, caption_inv))
                stats['attr_bag'] += 1

            # 2.3 混合变化 (Mixed Change)
            # 条件: 异衣、异包
            mixed_candidates = [s for s in sequences if s['parsed_attr'][0] != ref_cloth and s['parsed_attr'][1] != ref_bag]
            if mixed_candidates:
                tar = random.choice(mixed_candidates)
                
                # 构建正向任务列表
                fwd_tasks = ["change_cloth"]
                is_add = (not ref_bag) and tar['parsed_attr'][1]
                fwd_tasks.append("change_bag_add" if is_add else "change_bag_remove")
                
                # 构建反向任务列表
                rev_tasks = ["change_cloth"]
                rev_tasks.append("change_bag_remove" if is_add else "change_bag_add") # 反转包的操作
                
                # 生成组合指令
                caption = build_composite_instruction(templates, fwd_tasks)
                caption_inv = build_composite_instruction(templates, rev_tasks)
                
                triplets.append(create_entry(sid, ref, tar, "attribute_change", caption, caption_inv))
                stats['attr_mixed'] += 1

    # 4. 保存结果
    random.shuffle(triplets)
    print("Statistics:", json.dumps(stats, indent=2))
    print(f"Total Triplets: {len(triplets)}")
    
    with open(output_path, 'w') as f:
        json.dump(triplets, f, indent=4)
    print(f"Saved to {output_path}")

def create_entry(sid, ref_meta, tar_meta, task_type, caption, caption_inv):
    """
    构建符合 CASIA-B CIR 格式的 JSON 条目
    """
    return {
        "sid": sid,
        "dataset": "CCPG",
        "task": task_type,
        "caption": caption.replace("{subject}", "the person").replace("{view}", "another view"), # 简单后处理，防止占位符残留
        "caption_inv": caption_inv.replace("{subject}", "the person").replace("{view}", "another view"),
        "ref": {
            "sid": sid,
            "condition": ref_meta.get('condition', ref_meta.get('status')),
            "view": ref_meta.get('view'),
            "seq_path": ref_meta.get('seq_path', ref_meta.get('file_path')),
            "static_caption": ref_meta.get('static_caption', "A person walking.")
        },
        "tar": {
            "sid": sid,
            "condition": tar_meta.get('condition', tar_meta.get('status')),
            "view": tar_meta.get('view'),
            "seq_path": tar_meta.get('seq_path', tar_meta.get('file_path')),
            "static_caption": tar_meta.get('static_caption', "A person walking.")
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 请根据实际路径修改
    parser.add_argument('--meta_path', default='../CCPG_RGB_JSON/meta_ccpg_static.json') 
    parser.add_argument('--template_path', default='../CCPG_RGB_JSON/templates_instruction.json')
    parser.add_argument('--output', default='../CCPG_RGB_JSON/CCPG/ccpg_cir_final.json')
    args = parser.parse_args()

    build_ccpg_cir_json(args.meta_path, args.template_path, args.output)