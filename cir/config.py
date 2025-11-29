import os
import torch

# ==============================================================================
# 1. 路径配置 (Path Configuration)
# ==============================================================================
# 获取 config.py 当前所在的目录 (.../cir/)
_CONF_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../GaitCIR/)
PROJECT_ROOT = os.path.dirname(_CONF_DIR)

# 数据集根目录 (根据你的 AutoDL 环境修改)
DATASET_ROOT = '/root/autodl-tmp/CASIA-B-Processed'

# JSON 索引文件路径
# 注意：这里我保留了你的结构，指向 datasets/GaitCIR_RGB_JSON
JSON_ROOT = os.path.join(PROJECT_ROOT, 'datasets/CASIA-B_RGB_JSON')
TRAIN_JSON = os.path.join(JSON_ROOT, 'CASIA-B/casiab_cir_final.json')
SPLIT_CONFIG = os.path.join(JSON_ROOT, 'CASIA-B/CASIA-B.json')

# 输出目录 (Logs, Checkpoints)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'cir//checkpoint/MLP_Unmasked_alpha0.5_cos')

# ==============================================================================
# 2. 特征缓存配置 (Feature Caching)
# ==============================================================================
# 预提取特征的存储位置
FEATURE_ROOT_MASKED = os.path.join(DATASET_ROOT, 'CLIP_feature_Masked')
FEATURE_ROOT_UNMASKED = os.path.join(DATASET_ROOT, 'CLIP_feature_Unmasked')

# 【核心开关】默认使用的特征路径
# main.py 中的 --unmasked 参数可以覆盖这里的设置
FEATURE_ROOT = FEATURE_ROOT_MASKED 

# 默认模式开关
USE_FEATURES = True   # 默认读取 .pt 特征 (速度快)
USE_MASK = False       # 默认是 Masked 模式 (仅用于 Log 或 读图模式下的预处理)

# ==============================================================================
# 3. 模型与计算配置 (Model & Hardware)
# ==============================================================================
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DDP 相关配置
# 注意：OpenGait 通常根据 GPU 数量自动缩放 LR，这里我们先手动定死
BATCH_SIZE = 512       # 单卡 Batch Size (4卡的话总 Batch 就是 32*4=128)
NUM_WORKERS = 16       # 数据加载进程数

# ==============================================================================
# 4. 训练超参数 (Training Hyperparameters)
# ==============================================================================
# 学习率与优化器
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
WARMUP_STEPS = 100

# 采样配置 (GaitSet 风格)
TRAIN_MAX_FRAMES = 16  # 训练时：随机采样 30 帧进行聚合
TEST_MAX_FRAMES = all   # 测试时：使用 N 帧 (或全部) 获得更稳健特征

# ==============================================================================
# 5. 辅助配置 (Misc)
# ==============================================================================
# 粗粒度视角映射 (用于 compute_hierarchical_metrics 评估)
COARSE_VIEW_MAP = {
    "000": "front",
    "018": "front-side", "036": "front-side", "054": "front-side",
    "072": "side",       "090": "side",       "108": "side",
    "126": "back-side",  "144": "back-side",  "162": "back-side",
    "180": "back"
}

# Alpha: 逆向 Loss 的权重 (通常设为 0.1 ~ 0.5，不宜过大)
LOSS_ALPHA = 0.5 

# Type: 逆向 Loss 的类型
# 'nce'  = InfoNCE (对比学习，更强，计算量稍大) -> 你的原始版本
# 'cos'  = Cosine Similarity (简单相似度，更轻量，适合做正则项) -> 你建议的版本
LOSS_INV_TYPE = 'cos'