import torch
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from dataset_loader import GaitCIRDataset
import matplotlib.pyplot as plt

# 模拟 CLIP 的预处理 (不带 Normalize 以便可视化)
simple_transform = Compose([
    Resize(224, interpolation=3),
    CenterCrop(224),
    ToTensor(),
])

def test():
    print("🚀 开始 DataLoader 冒烟测试...")
    
    # ================= 配置区域 =================
    MASTER_JSON = '../../datasets/GaitCIR_RGB/casiab_cir_final.json'
    SPLIT_CONFIG = '../../datasets/GaitCIR_RGB/Split/CASIA-B.json'
    MODE = 'train' # 测试训练集数据
    
    # 初始化 Dataset
    dataset = GaitCIRDataset(
        json_path=MASTER_JSON,
        data_root='../../datasets/CASIA-B-Processed',
        split_config_path=SPLIT_CONFIG, # 传入分割配置
        mode=MODE,                      # 指定模式
        max_frames=1,                   # 训练模式下只采单帧
        transform=simple_transform,
        subject_token="the person",
        return_static=True              # 必须为 True 才能打印静态描述
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 读取一个 Batch
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"❌ DataLoader 读取失败: {e}")
        return

    # --- 打印调试信息 ---
    print("\n" + "="*40)
    print(f"🔍 Batch Keys: {list(batch.keys())}")
    
    # 检查形状
    # 训练模式下应该是 [4, 3, 224, 224]
    # 测试模式下应该是 [4, 8, 3, 224, 224] (List of Tensors 或 Stacked Tensor)
    ref_data = batch['ref_imgs']
    if isinstance(ref_data, list):
        print(f"🖼️ Ref Image (List): Length {len(ref_data)}, Item Shape {ref_data[0].shape}")
        #如果是列表取第一帧用于可视化
        ref_tensor = ref_data[0]
        tar_tensor = batch['tar_imgs'][0]
    else:
        print(f"🖼️ Ref Image (Tensor): Shape {ref_data.shape}")
        ref_tensor = batch['ref_imgs'][0]
        tar_tensor = batch['tar_imgs'][0]

    print("-" * 40)
    
    # 打印文本 (检查占位符替换)
    print(f"📝 Instruction: {batch['text'][0]}")
    print(f"📝 Instruction_inv: {batch['text_inv'][0]}")
    
    # 检查静态描述是否存在
    if 'ref_text' in batch:
        print(f"🏷️ Ref Static:  {batch['ref_text'][0]}")
        print(f"🏷️ Tar Static:  {batch['tar_text'][0]}")
    else:
        print("⚠️ Warning: 'ref_text' not found. Did you set return_static=True?")
        
    print(f"📌 Task Type:   {batch['task'][0]}")
    print(f"🆔 Subject ID:  {batch.get('sid', 'N/A')[0]}")
    print(f"🎨 Condition:   {batch.get('cond', 'N/A')[0]}")
    print("-" * 40)
    
    # --- 可视化检查 ---
    # Tensor (C, H, W) -> Numpy (H, W, C)
    ref_img = ref_tensor.permute(1, 2, 0).numpy()
    tar_img = tar_tensor.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Ref: {batch['text'][0][:30]}...")
    plt.imshow(ref_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Target\n(Should match instruction)")
    plt.imshow(tar_img)
    plt.axis('off')
    
    save_path = "loader_check.png"
    plt.savefig(save_path)
    print(f"✅ 可视化结果已保存至 {save_path}")
    print("   -> 请检查背景是否为全黑 (Masked RGB)")
    print("   -> 请检查 Ref 和 Tar 是否符合文本描述")

if __name__ == '__main__':
    test()