# test_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from dataset_loader import GaitCIRDataset
import matplotlib.pyplot as plt

# 模拟 CLIP 的预处理
simple_transform = Compose([
    Resize(224, interpolation=3),
    CenterCrop(224),
    ToTensor(),
    # Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.276)) # 暂时注释掉以便可视化
])

def test():
    # 初始化 Dataset
    dataset = GaitCIRDataset(
        json_path='../../datasets/GaitCIR_RGB/casiab_cir_train_split.json',
        data_root='../../datasets/CASIA-B-Processed',
        transform=simple_transform,
        subject_token="the person",
        return_static=True
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 读取一个 Batch
    batch = next(iter(loader))
    
    print(f"Keys: {batch.keys()}")
    print(f"Ref Image Shape: {batch['ref_img'].shape}") # Should be [4, 3, 224, 224]
    print(f"Text Example: {batch['text'][0]}")
    
    # 可视化检查 (保存第一张图)
    ref_tensor = batch['ref_img'][0]
    tar_tensor = batch['tar_img'][0]
    
    # Tensor -> Numpy (C, H, W) -> (H, W, C)
    ref_img = ref_tensor.permute(1, 2, 0).numpy()
    tar_img = tar_tensor.permute(1, 2, 0).numpy()
    
    # 检查背景是否全黑 (Masked RGB 验证)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Ref: {batch['text'][0][:20]}...")
    plt.imshow(ref_img)
    
    plt.subplot(1, 2, 2)
    plt.title("Target")
    plt.imshow(tar_img)
    
    plt.savefig("loader_check.png")
    print("-" * 30)
    print(f"🔍 Batch Keys: {list(batch.keys())}") # 用 list() 包一下更整洁
    print(f"🖼️ Ref Image Shape: {batch['ref_img'].shape}") 
    print("-" * 30)
    
    # 打印完整的文本，不要截断，方便检查 {subject} 是否替换成功
    print(f"📝 Instruction: {batch['text'][0]}")
    print(f"🏷️ Ref Static:  {batch['ref_text'][0]}")
    print(f"🏷️ Tar Static:  {batch['tar_text'][0]}")
    print(f"📌 Task Type:  {batch['task'][0]}") # 看看这是个什么任务
    print("-" * 30)
    print("可视化结果已保存至 loader_check.png，请检查背景是否为黑色！")

if __name__ == '__main__':
    test()