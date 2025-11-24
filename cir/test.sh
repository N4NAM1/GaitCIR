#!/bin/bash

# ==================================================================
# GaitCIR Testing Script (Comprehensive)
# ==================================================================
# ⚠️ 注意事项:
# 1. 检索任务建议使用单卡 (CUDA_VISIBLE_DEVICES=0) 以保证指标绝对准确。
# 2. 请确保 --ckpt 指向的模型权重与测试模式匹配！
#    (例如: 不要用 Masked 训练的权重去测 Unmasked 数据，除非你想做鲁棒性实验)
# ==================================================================

# Set GPU (Default: Use GPU 0)
GPU=0

# ==================================================================
# 🧪 实验 A: 标准特征模式 (Masked / Default)
# 对应: bash train.sh (不加参数)
# ==================================================================
# echo "🚀 [Test A] Testing Masked Features (Standard)..."
# CUDA_VISIBLE_DEVICES=$GPU python cir/main.py \
#     --phase test \
#     --ckpt cir/checkpoints/combiner_ep30.pth \
#     # --local_rank 0  <-- 单卡 python 启动不需要这个，torchrun 才需要


# ==================================================================
# 🧪 实验 B: 背景模式 (Unmasked)
# 对应: bash train.sh ... --unmasked
# ==================================================================
# 如果你训练了 Unmasked 模型，请取消下面注释来测试
echo "🚀 [Test B] Testing Unmasked Features (Background)..."
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --phase test \
    --unmasked \
    --ckpt /root/work/GaitCIR/cir/checkpoint/MLP_Unmasked_alpha0.5_cos/combiner_ep15.pth


# ==================================================================
# 🧪 实验 C: 跨域鲁棒性测试 (Cross-Domain Robustness)
# 有趣的实验: 用 Masked 训练的模型，去测 Unmasked 数据
# 看看模型是否真的学会了忽略背景？
# ==================================================================
# echo "🚀 [Test C] Cross-Domain: Masked Model -> Unmasked Data..."
# CUDA_VISIBLE_DEVICES=$GPU python cir/main.py \
#     --phase test \
#     --unmasked \
#     --ckpt cir/checkpoints/combiner_ep30.pth


# ==================================================================
# 🧪 实验 D: 原始图像模式 (Image Mode)
# 对应: bash train.sh ... --no_feat
# ==================================================================
# 适用于没有提取特征文件，直接读图测试
# echo "🚀 [Test D] Testing Raw Images (Slow)..."
# CUDA_VISIBLE_DEVICES=$GPU python cir/main.py \
#     --phase test \
#     --no_feat \
#     --ckpt cir/checkpoints/combiner_ep30.pth