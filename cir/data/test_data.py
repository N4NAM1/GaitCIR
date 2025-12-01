import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import numpy as np

# 假设你的 Dataset 文件名为 dataset_loader.py
from dataset_loader import GaitCIRDataset

# 模拟预处理
simple_transform = Compose([
    Resize(224, interpolation=3),
    CenterCrop(224),
    ToTensor(),
])

def test():
    print("🚀 开始 DataLoader 通用冒烟测试...")

    # ================= ⚙️ 数据集配置区域 =================
    # 你可以在这里修改路径，然后通过修改 CURRENT_DATASET 变量来切换
    
    DATASET_CONFIGS = {
        "CASIA-B": {
            "ROOT": "/root/autodl-tmp/CASIA-B-Processed",
            "JSON": "/root/work/GaitCIR/datasets/CASIA-B_RGB_JSON/CASIA-B/casiab_cir_final.json",
            "NAME": "CASIA-B"
        },
        "CCPG": {
            "ROOT": "/root/autodl-tmp/CCPG_Processed",
            "JSON": "/root/work/GaitCIR/datasets/CCPG_RGB_JSON/CCPG/ccpg_cir_final.json", # 请替换真实路径
            "NAME": "CCPG"
        },
        "SUSTech1K": {
            "ROOT": "/root/autodl-tmp/SUSTech1K_Processed",
            "JSON": "path/to/sustech1k_cir_final.json", # 请替换真实路径
            "NAME": "SUSTech1K"
        }
    }

    # 🔥 在这里切换你要测试的数据集！
    CURRENT_DATASET = "CASIA-B"  # 选项: "CASIA-B", "CCPG", "SUSTech1K"
 # =======================================================

    cfg = DATASET_CONFIGS[CURRENT_DATASET]
    print(f"📂 当前测试目标: {cfg['NAME']}")

    # 1. 初始化 Dataset
    try:
        dataset = GaitCIRDataset(
            json_path=cfg["JSON"],
            data_root=cfg["ROOT"],
            dataset_name=cfg["NAME"],
            mode='train',
            max_frames=16,                   
            transform=simple_transform,
            subject_token="the person",
            return_static=True,
            use_mask=False,                 
            use_features=False
        )
    except Exception as e:
        print(f"\n❌ Dataset 初始化失败！错误信息:\n{e}")
        return

    print(f"✅ Dataset 加载成功，数据总量: {len(dataset)}")
    
    # 2. 初始化 DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. 读取一个 Batch
    print("⏳ 正在读取第一个 Batch...")
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"\n❌ DataLoader 读取失败！错误信息:\n{e}")
        return

    # --- 打印详细信息 ---
    print("\n" + "="*40)
    
    # 提取 Ref 和 Tar 数据
    ref_data = batch['ref_imgs']
    tar_data = batch['tar_imgs'] # 🔥 新增 Target 读取
    
    print(f"🖼️ Ref Shape: {ref_data.shape}") 
    print(f"🖼️ Tar Shape: {tar_data.shape}") # 🔥 打印 Target 形状
    
    # 处理数据维度用于可视化 (取出 Batch 0, Frame 0)
    def get_first_img(tensor_data):
        if tensor_data.dim() == 5: # [B, T, C, H, W]
            return tensor_data[0][0]
        else: # [B, C, H, W]
            return tensor_data[0]

    ref_tensor = get_first_img(ref_data)
    tar_tensor = get_first_img(tar_data)

    # 打印文本和元数据
    sid = batch['sid'][0]
    view = batch['view'][0]
    cond = batch['cond'][0]
    text = batch['text'][0]
    
    print(f"📌 Subject ID: {sid}")
    print(f"📌 View Angle: {view}")
    print(f"📌 Condition:  {cond}")
    print(f"📝 Instruction: {text}")
    print("-" * 40)

    # --- 可视化对比 (Ref vs Target) ---
    ref_np = ref_tensor.permute(1, 2, 0).numpy()
    tar_np = tar_tensor.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 5))
    
    # 绘制 Reference
    plt.subplot(1, 2, 1)
    plt.imshow(ref_np)
    plt.title(f"Reference Image\n{text[:20]}...") # 显示部分指令
    plt.axis('off')

    # 绘制 Target
    plt.subplot(1, 2, 2)
    plt.imshow(tar_np)
    plt.title(f"Target Image\nID: {sid} | View: {view}")
    plt.axis('off')
    
    save_path = f"check_{cfg['NAME']}_pair.png"
    plt.savefig(save_path)
    print(f"✅ 可视化对比图已保存至: {save_path}")
    print("👀 请检查：Ref 和 Target 是否看起来是同一个人？(Identity Consistency)")

if __name__ == '__main__':
    test()