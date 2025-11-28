import os
import json
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = './CASIA-B-Processed/RGB'
OUTPUT_META = './GaitCIR_RGB/meta_casiab_static.json'

# ================= 1. 静态描述词典 =================
# 用于生成 static_caption
STATE_DESC_POOL = {
    "nm": [
        "walking normally", "walking in a standard gait", "moving naturally",
        "walking empty-handed", "carrying nothing", "without any bags", 
        "walking unencumbered", "with hands free", "holding nothing",
        "in standard clothing", "dressed casually", "wearing plain clothes",
        "in a regular outfit", "wearing everyday attire"
    ],
    "bg": [
        "carrying a backpack", "wearing a backpack", "with a backpack on", 
        "strapped with a knapsack", "carrying a rucksack",
        "carrying a bag", "holding a bag", "toting a handbag", 
        "with a bag slung over the shoulder", "carrying a satchel",
        "holding a briefcase", "lugging a bag",
    ],
    "cl": [
        "wearing a thick coat", "dressed in a heavy coat", "wearing a long coat",
        "bundled up in a coat", "wearing an overcoat",
        "wearing a down jacket", "in winter clothing", "dressed for cold weather",
        "wearing a parka", "in a puffy jacket", "wearing heavy outerwear",
        "looking bulky in a coat", "wearing warm clothes"
    ]
}

VIEW_DESC_POOL = {
    "000": [
        "a front view", "a frontal angle", "a 0-degree view", "a face-to-face view",
        "a view facing the camera", 
        "a head-on view", "a straight-on shot", "a view facing forward"
    ],
    "018": [
        "a front-side view", "an oblique angle"
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
        "a slight profile", "a near-side view"
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
# ================= 2. 核心生成逻辑 =================

def generate_static_caption(action, view_text):
    rand = random.random()
    if rand < 0.4:
        return f"{{subject}} {action} viewed from the {view_text}."
    elif rand < 0.7:
        return f"A {view_text} of {{subject}} {action}."
    else:
        return f"{{subject}}, {action}, {view_text}."

def generate_meta():
    print(f"正在扫描并生成元数据: {DATASET_ROOT} ...")
    metadata = []
    
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 错误: 数据集路径不存在 {DATASET_ROOT}")
        return

    subjects = sorted(os.listdir(DATASET_ROOT))
    
    for sid in tqdm(subjects):
        sid_path = os.path.join(DATASET_ROOT, sid)
        if not os.path.isdir(sid_path): continue
        
        types = sorted(os.listdir(sid_path))
        for type_str in types: 
            cond = type_str.split('-')[0] 
            type_path = os.path.join(sid_path, type_str)
            if not os.path.isdir(type_path): continue
            
            views = sorted(os.listdir(type_path))
            for view in views:
                view_path = os.path.join(type_path, view)
                if not os.path.exists(view_path) or not os.listdir(view_path): continue
                
                # 生成描述
                action_text = random.choice(STATE_DESC_POOL.get(cond, ["walking"]))
                view_text = random.choice(VIEW_DESC_POOL.get(view, ["view"]))
                static_cap = generate_static_caption(action_text, view_text)
                
                item = {
                    "sid": sid,
                    "condition": cond,
                    "view": view,
                    "seq_path": os.path.join(sid, type_str, view), # seq_path 依然作为唯一标识
                    "static_caption": static_cap 
                }
                metadata.append(item)
                
    with open(OUTPUT_META, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ 静态元数据生成完毕! ")

if __name__ == '__main__':
    generate_meta()