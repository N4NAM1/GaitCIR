import os
import json
import argparse
from tqdm import tqdm

def generate_ccpg_meta_aligned(data_root, output_path):
    """
    生成与 CASIA-B 格式完全一致的 CCPG 元数据。
    
    Target Format:
    {
        "unique_key": {
            "sid": "001",
            "condition": "U0_D0_BG",
            "view": "01",
            "seq_path": "001/U0_D0_BG/01_0/01_0-aligned-rgbs.pkl",
            "static_caption": "A person with a bag ..."
        }
    }
    """
    meta_data = {}
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    subjects = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print(f"Found {len(subjects)} subjects in {data_root}")

    count = 0
    for pid in tqdm(subjects, desc="Indexing CCPG"):
        pid_path = os.path.join(data_root, pid)
        if not os.path.isdir(pid_path): continue
        
        statuses = sorted(os.listdir(pid_path))
        
        for status in statuses:
            status_path = os.path.join(pid_path, status)
            if not os.path.isdir(status_path): continue
            
            view_seqs = sorted(os.listdir(status_path))
            
            for vs_folder in view_seqs:
                if '_' not in vs_folder: continue
                try:
                    view_code, seq_num = vs_folder.split('_')
                except ValueError:
                    continue

                # 构造文件名
                pkl_name = f"{vs_folder}-aligned-rgbs.pkl"
                abs_file = os.path.join(status_path, vs_folder, pkl_name)
                
                if os.path.exists(abs_file):
                    # === 构造与 CASIA-B 一致的字段 ===
                    
                    # 1. 相对路径 (seq_path)
                    rel_path = os.path.join(pid, status, vs_folder, pkl_name)
                    
                    # 2. 简单的静态描述规则 (粗粒度)
                    # 虽然我们未来要用 MLLM，但现在先填一个占位符保证代码能跑
                    base_cap = "{{Subject}} walking"
                    if "BG" in status:
                        base_cap += " carrying a bag"
                    if "U" in status and "U0" not in status:
                        base_cap += ", wearing a different upper outfit"
                    if "D" in status and "D0" not in status:
                        base_cap += ", wearing different pants"
                    base_cap += "."
                    
                    # 3. 构造 Entry
                    # Key 依然建议保持唯一性
                    key = f"{pid}_{status}_{view_code}_{seq_num}"
                    
                    meta_data[key] = {
                        "sid": pid,               # 对应 CASIA-B: sids
                        "condition": status,      # 对应 CASIA-B: condition
                        "view": view_code,        # 对应 CASIA-B: view
                        "seq_path": rel_path,     # 对应 CASIA-B: seq_path
                        "static_caption": base_cap # 对应 CASIA-B: static_caption
                    }
                    count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    print(f"Saved aligned meta data to {output_path}. Total: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='CCPG_Processed absolute path')
    parser.add_argument('--output', type=str, default='datasets/CCPG/meta_ccpg.json')
    args = parser.parse_args()
    
    generate_ccpg_meta_aligned(args.data_root, args.output)