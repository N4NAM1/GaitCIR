import os
import json
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = './CASIA-B-Processed/RGB'
OUTPUT_META = './GaitCIR_RGB/meta_casiab_static.json'

# ================= 1. 多样化描述词典 =================

# 状态描述池 (State Diversity)
# 注意：这里的描述是"状态补充"，紧跟在主语后面
STATE_DESC_POOL = {
    "nm": [
        "walking normally", 
        "walking without carrying anything", 
        "in standard clothing", 
        "walking empty-handed", 
        "carrying nothing",
        "with a normal gait"
    ],
    "bg": [
        "carrying a bag", 
        "carrying a backpack", 
        "holding a bag", 
        "with a bag", 
        "wearing a backpack", 
        "carrying luggage"
    ],
    "cl": [
        "wearing a thick coat", 
        "dressed in a heavy coat", 
        "wearing a down jacket", 
        "in winter clothing", 
        "wearing an outer garment", 
        "clad in a parka"
    ]
}

# 视角描述池 (View Diversity)
VIEW_DESC_POOL = {
    "000": ["front view", "frontal view", "view from the front", "head-on view"],
    "018": ["front-side view", "slight oblique view", "anterolateral view"],
    "036": ["front-side view", "oblique view", "angle from the front-right"],
    "054": ["front-side view", "oblique perspective", "quarter view"],
    "072": ["side view", "profile view", "lateral view"],
    "090": ["side view", "90-degree profile", "view from the side", "full profile"],
    "108": ["side view", "lateral angle", "profile from back"],
    "126": ["back-side view", "rear-oblique view"],
    "144": ["back-side view", "oblique view from behind"],
    "162": ["back-side view", "posterolateral view"],
    "180": ["back view", "rear view", "view from behind", "back-facing angle"]
}

# ================= 2. 核心生成逻辑 =================

def generate_static_caption(action, view_text):
    """
    随机组合句式结构，且必须包含 {subject}
    """
    rand = random.random()
    
    if rand < 0.4:
        # 标准句式: {subject} [action] viewed from the [view].
        # 例如: {subject} walking normally viewed from the side view.
        return f"{{subject}} {action} viewed from the {view_text}."
        
    elif rand < 0.7:
        # 倒装句式: A [view] of {subject} [action].
        # 例如: A side view of {subject} walking normally.
        # 注意：这里 "A ... view" 是对图片的描述，不算是代词，但 subject 必须是占位符
        return f"A {view_text} of {{subject}} {action}."
        
    else:
        # 简洁/状态导向: {subject}, [Action], [View].
        # 例如: {subject}, carrying a bag, side view.
        return f"{{subject}}, {action}, {view_text}."

def generate_meta():
    print(f"正在扫描并生成元数据 (含占位符): {DATASET_ROOT} ...")
    metadata = []
    
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
                if not os.listdir(view_path): continue
                
                # === 核心：随机抽取描述 ===
                action_text = random.choice(STATE_DESC_POOL.get(cond, ["walking"]))
                view_text = random.choice(VIEW_DESC_POOL.get(view, ["specific angle"]))
                
                # === 生成带占位符的静态描述 ===
                # 注意：这里的 static_caption 现在是 "{subject} walking..."
                static_caption = generate_static_caption(action_text, view_text)
                
                item = {
                    "sid": sid,
                    "condition": cond,
                    "type_id": type_str,
                    "view": view,
                    "view_text": view_text, 
                    "seq_path": os.path.join(sid, type_str, view),
                    "static_caption": static_caption 
                }
                metadata.append(item)
                
    with open(OUTPUT_META, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ 静态元数据生成完毕! (已集成 {{subject}} 占位符)")

if __name__ == '__main__':
    generate_meta()