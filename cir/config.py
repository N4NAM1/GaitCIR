import os
import torch

# === 路径配置 ===
PROJECT_ROOT = os.path.dirname('../') # 根目录
DATASET_ROOT = os.path.join(PROJECT_ROOT, '../../autodl-tmp/CASIA-B-Processed')
JSON_ROOT = os.path.join(PROJECT_ROOT, 'datasets/GaitCIR_RGB_JSON')

TRAIN_JSON = os.path.join(JSON_ROOT, 'CASIA-B/casiab_cir_final.json')
SPLIT_CONFIG = os.path.join(JSON_ROOT, 'CASIA-B/CASIA-B.json')

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'cir/checkpoints')

# === 训练配置 ===
MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 粗粒度视角映射 (评估用) ===
COARSE_VIEW_MAP = {
    "000": "front",
    "018": "front-side", "036": "front-side", "054": "front-side",
    "072": "side",       "090": "side",       "108": "side",
    "126": "back-side",  "144": "back-side",  "162": "back-side",
    "180": "back"
}