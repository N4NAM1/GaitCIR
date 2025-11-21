import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm

# ================= 加速配置区域 =================
INPUT_ROOT = './DatasetB-1/video' 
OUTPUT_ROOT = './CASIA-B-Processed'
MODEL_NAME = 'yolo11x-seg.pt' 


BATCH_SIZE = 32     # 每次处理的帧数，显存允许越大越好

TARGET_CLASSES = [0, 24, 26, 28] # 人, 背包, 手提包, 箱子
TARGET_SIZE = 224  #最终输出的图像尺寸
CONF_THRESHOLD = 0.4    # 置信度阈值
BORDER_MARGIN = 10  # 边缘触碰判定的边距
# ==============================================

def make_square_padding(img, target_size=224):
    """保持长宽比，补黑边成正方形"""
    h, w = img.shape[:2]
    longest_edge = max(h, w)
    top = (longest_edge - h) // 2
    bottom = longest_edge - h - top
    left = (longest_edge - w) // 2
    right = longest_edge - w - left
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(img_padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

def parse_casiab_filename(filename):
    """解析 CASIA-B 视频文件名，返回 subject_id, type_str, view"""
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('-')
    if len(parts) < 4: return None, None, None
    return parts[0], f"{parts[1]}-{parts[2]}", parts[3]

def calculate_iou_or_intersection(boxA, boxB):
    """计算两个边界框是否有交集"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (max(0, xB - xA) * max(0, yB - yA)) > 0

def is_touching_border(box, img_w, img_h, margin=10):
    """判断边界框是否触碰图像边缘"""
    x1, y1, x2, y2 = box
    if x1 < margin: return True
    if x2 > img_w - margin: return True
    if y1 < margin: return True
    if y2 > img_h - margin: return True
    return False

def expand_box(box, img_w, img_h, ratio_w=0.05, ratio_h=0.05):
    """外扩边界框，防止切掉头脚"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    dx = int(w * ratio_w)
    dy = int(h * ratio_h)
    new_x1 = max(0, x1 - dx)
    new_y1 = max(0, y1 - dy)
    new_x2 = min(img_w, x2 + dx)
    new_y2 = min(img_h, y2 + dy)
    return (new_x1, new_y1, new_x2, new_y2)

def process_batch(model, frames, frame_indices, subject_id, type_str, view, rgb_dir, mask_dir, frame_w, frame_h):
    """处理一个 Batch 的帧"""
    
    # 批量推理
    results = model(frames, classes=TARGET_CLASSES, conf=CONF_THRESHOLD, verbose=False, retina_masks=True)
    
    for i, result in enumerate(results):
        frame = frames[i]
        frame_idx = frame_indices[i]
        
        if len(result.boxes) > 0 and result.masks is not None:
            boxes = result.boxes.data.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            
            # 1. 找主角 (Person class=0)
            person_indices = [j for j, box in enumerate(boxes) if int(box[5]) == 0]
            if not person_indices: continue
            
            best_person_idx = -1
            max_area = 0
            for idx in person_indices:
                area = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                if area > max_area:
                    max_area = area
                    best_person_idx = idx
            
            # 2. 基础融合逻辑
            main_person_box = boxes[best_person_idx][:4]
            combined_mask = masks[best_person_idx]
            
            # 尺寸对齐
            if combined_mask.shape[:2] != frame.shape[:2]:
                combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask_binary = (combined_mask > 0.5).astype(np.uint8)

            # 3. 融合背包 (Class 24,26,28 或 其他Person碎片)
            other_indices = [j for j, box in enumerate(boxes) if j != best_person_idx]
            for idx in other_indices:
                obj_box = boxes[idx][:4]
                if calculate_iou_or_intersection(main_person_box, obj_box):
                    obj_mask = masks[idx]
                    if obj_mask.shape[:2] != frame.shape[:2]:
                        obj_mask = cv2.resize(obj_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask_binary = cv2.bitwise_or(combined_mask_binary, (obj_mask > 0.5).astype(np.uint8))
            
            # 4. 计算最终包围盒
            coords = cv2.findNonZero(combined_mask_binary)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                x1, y1, x2, y2 = x, y, x+w, y+h
                
                # 5. 边缘过滤 (触碰边缘则丢弃)
                if is_touching_border((x1, y1, x2, y2), frame_w, frame_h, margin=BORDER_MARGIN):
                    continue

                # 6. 框体外扩 (防止切头)
                x1, y1, x2, y2 = expand_box((x1, y1, x2, y2), frame_w, frame_h, ratio_w=0.05, ratio_h=0.1)
                
                # 7. 裁剪与保存
                mask_final = combined_mask_binary * 255 
                crop_rgb = frame[y1:y2, x1:x2]
                crop_mask = mask_final[y1:y2, x1:x2]
                
                if crop_rgb.size > 0:
                    final_rgb = make_square_padding(crop_rgb, TARGET_SIZE)
                    final_mask = make_square_padding(crop_mask, TARGET_SIZE)
                    
                    filename = f"{subject_id}-{type_str}-{view}-{frame_idx:03d}"
                    # 保存 (JPG存RGB, PNG存Mask)
                    cv2.imwrite(os.path.join(rgb_dir, filename + ".jpg"), final_rgb)
                    cv2.imwrite(os.path.join(mask_dir, filename + ".png"), final_mask)

def process_casiab_fast():
    # 准备目录
    rgb_root = os.path.join(OUTPUT_ROOT, 'RGB')
    mask_root = os.path.join(OUTPUT_ROOT, 'Mask')
    if not os.path.exists(rgb_root): os.makedirs(rgb_root)
    if not os.path.exists(mask_root): os.makedirs(mask_root)

    print(f"加载模型: {MODEL_NAME} | 启用 Batch 加速 (Size={BATCH_SIZE})...")
    model = YOLO(MODEL_NAME)

    video_files = [f for f in os.listdir(INPUT_ROOT) if f.endswith(('.avi', '.mp4'))]

    for video_file in tqdm(video_files, desc="Processing"):
        subject_id, type_str, view = parse_casiab_filename(video_file)
        if subject_id is None: continue
            
        curr_rgb_dir = os.path.join(rgb_root, subject_id, type_str, view)
        curr_mask_dir = os.path.join(mask_root, subject_id, type_str, view)
        if not os.path.exists(curr_rgb_dir): os.makedirs(curr_rgb_dir)
        if not os.path.exists(curr_mask_dir): os.makedirs(curr_mask_dir)
        
        cap = cv2.VideoCapture(os.path.join(INPUT_ROOT, video_file))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 1 
        batch_frames = []
        batch_indices = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            
            # 缓存满则处理
            if len(batch_frames) == BATCH_SIZE:
                process_batch(model, batch_frames, batch_indices, subject_id, type_str, view, curr_rgb_dir, curr_mask_dir, frame_w, frame_h)
                batch_frames = []
                batch_indices = []
            
            frame_idx += 1
        
        # 处理尾巴
        if len(batch_frames) > 0:
            process_batch(model, batch_frames, batch_indices, subject_id, type_str, view, curr_rgb_dir, curr_mask_dir, frame_w, frame_h)
            
        cap.release()

if __name__ == '__main__':
    process_casiab_fast()