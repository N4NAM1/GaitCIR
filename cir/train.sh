# ==================================================================
# GaitCIR Training Script (OpenGait Style)
# ==================================================================
# 说明:
# 1. nproc_per_node 应等于你的显卡数量 (例如 0,1,2,3 是 4 张卡)
# 2. --use_env 推荐加上，它会让 PyTorch 通过环境变量传递 Rank 信息，更稳定
# ==================================================================

# --- 实验 A: 极速特征模式 (Masked, 默认) ---
# 最快，显存占用最小，适合快速调参
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --master_port=12345 \
#     --use_env \
#     main.py --phase train

# --- 实验 B: 背景消融实验 (Unmasked 特征) ---
# 探究背景噪声的影响，需确保 config.py 中 FEATURE_ROOT_UNMASKED 路径正确
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12346 \
    --use_env \
    main.py --phase train --unmasked

# --- 实验 C: 原始图像模式 (Image Mode) ---
# 读取原始 JPG 图片，速度较慢，用于 Debug 或可视化
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=12347 \
#     --use_env \
#     cir/main.py --phase train --no_feat