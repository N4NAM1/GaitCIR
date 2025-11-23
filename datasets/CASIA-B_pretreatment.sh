#!/bin/bash

# ================= 显卡设置 =================
# 例如使用 0,1,2,3 四张卡
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
# ================= 路径配置 =================
DATA_ROOT="/root/autodl-tmp"
CASIAB_INPUT="${DATA_ROOT}/CASIA-B/DatasetB-2/DatasetB-2/video"
CASIAB_OUTPUT="${DATA_ROOT}/CASIA-B-Processed"

# ================= 运行 =================
echo "🚀 Start Multi-GPU Processing..."

# 注意：
# --gpus "0,1,2,3"  表示同时使用这4张卡
# --workers 4       表示每张卡配 4 个 CPU 线程（如果4张卡，总共会有 16 个线程在跑）

python 01_extract_sil_rgb_yolo.py \
    --input_path "$CASIAB_INPUT" \
    --output_path "$CASIAB_OUTPUT" \
    --model "yolo11x-seg.pt" \
    --gpus "0" \
    --batch_size 64 \
    --workers 16 \
    --img_size 224 \
    --conf 0.5

echo "🎉 All Done!"