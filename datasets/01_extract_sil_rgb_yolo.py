import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import torch.multiprocessing as mp
import math
import time

# ================= 配置 =================
# 目标类别 (COCO): 0=Person, 24=Backpack, 26=Handbag, 28=Suitcase
TARGET_CLASSES = [0, 24, 26, 28] 
# =======================================

def get_args():
    parser = argparse.ArgumentParser(description="GaitCIR Preprocessing (Multi-GPU Supported)")
    
    # 路径参数
    parser.add_argument('--input_path', type=str, required=True, help='Raw video dataset root path')
    parser.add_argument('--output_path', type=str, required=True, help='Processed data output path')
    parser.add_argument('--model', type=str, default='yolo11x-seg.pt', help='YOLO model path')
    
    # 多卡参数
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use, e.g., "0,1,2,3"')
    
    # 性能参数
    parser.add_argument('--workers', type=int, default=4, help='Number of CPU threads PER GPU')
    parser.add_argument('--batch_size', type=int, default=32, help='Inference batch size')
    
    # 算法参数
    parser.add_argument('--img_size', type=int, default=224, help='Target image size')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--margin', type=int, default=10, help='Border margin')
    
    return parser.parse_args()

# ================== 辅助函数 (图像处理) ==================
def make_square_padding(img, target_size=224):
    h, w = img.shape[:2]
    longest_edge = max(h, w)
    top = (longest_edge - h) // 2
    bottom = longest_edge - h - top
    left = (longest_edge - w) // 2
    right = longest_edge - w - left
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(img_padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

def parse_casiab_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('-')
    if len(parts) < 4: return None, None, None
    return parts[0], f"{parts[1]}-{parts[2]}", parts[3]

def calculate_iou_or_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (max(0, xB - xA) * max(0, yB - yA)) > 0

def is_touching_border(box, img_w, img_h, margin=10):
    x1, y1, x2, y2 = box
    if x1 < margin: return True
    if x2 > img_w - margin: return True
    if y1 < margin: return True
    if y2 > img_h - margin: return True
    return False

def expand_box(box, img_w, img_h, ratio_w=0.05, ratio_h=0.05):
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

# ================== 核心处理逻辑 ==================

def process_batch(model, gpu_lock, device, frames, frame_indices, subject_id, type_str, view, rgb_dir, mask_dir, frame_w, frame_h, args):
    """处理一个 Batch 的帧"""
    with gpu_lock:
        results = model(frames, classes=TARGET_CLASSES, conf=args.conf, verbose=False, retina_masks=True, device=device)
    
    for i, result in enumerate(results):
        frame = frames[i]
        frame_idx = frame_indices[i]
        
        if len(result.boxes) > 0 and result.masks is not None:
            boxes = result.boxes.data.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            
            person_indices = [j for j, box in enumerate(boxes) if int(box[5]) == 0]
            if not person_indices: continue
            
            best_person_idx = -1
            max_area = 0
            for idx in person_indices:
                area = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                if area > max_area:
                    max_area = area
                    best_person_idx = idx
            
            main_person_box = boxes[best_person_idx][:4]
            combined_mask = masks[best_person_idx]
            
            if combined_mask.shape[:2] != frame.shape[:2]:
                combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask_binary = (combined_mask > 0.5).astype(np.uint8)

            other_indices = [j for j, box in enumerate(boxes) if j != best_person_idx]
            for idx in other_indices:
                obj_box = boxes[idx][:4]
                if calculate_iou_or_intersection(main_person_box, obj_box):
                    obj_mask = masks[idx]
                    if obj_mask.shape[:2] != frame.shape[:2]:
                        obj_mask = cv2.resize(obj_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask_binary = cv2.bitwise_or(combined_mask_binary, (obj_mask > 0.5).astype(np.uint8))
            
            coords = cv2.findNonZero(combined_mask_binary)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                x1, y1, x2, y2 = x, y, x+w, y+h
                
                if is_touching_border((x1, y1, x2, y2), frame_w, frame_h, margin=args.margin):
                    continue

                x1, y1, x2, y2 = expand_box((x1, y1, x2, y2), frame_w, frame_h, ratio_w=0.05, ratio_h=0.1)
                
                mask_final = combined_mask_binary * 255 
                crop_rgb = frame[y1:y2, x1:x2]
                crop_mask = mask_final[y1:y2, x1:x2]
                
                if crop_rgb.size > 0:
                    final_rgb = make_square_padding(crop_rgb, args.img_size)
                    final_mask = make_square_padding(crop_mask, args.img_size)
                    filename = f"{subject_id}-{type_str}-{view}-{frame_idx:03d}"
                    
                    cv2.imwrite(os.path.join(rgb_dir, filename + ".jpg"), final_rgb)
                    cv2.imwrite(os.path.join(mask_dir, filename + ".png"), final_mask)

def process_one_video(video_file, model, gpu_lock, device, rgb_root, mask_root, args):
    """单个视频处理流程"""
    try:
        subject_id, type_str, view = parse_casiab_filename(video_file)
        if subject_id is None: return
            
        curr_rgb_dir = os.path.join(rgb_root, subject_id, type_str, view)
        curr_mask_dir = os.path.join(mask_root, subject_id, type_str, view)
        
        os.makedirs(curr_rgb_dir, exist_ok=True)
        os.makedirs(curr_mask_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(os.path.join(args.input_path, video_file))
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
            
            if len(batch_frames) == args.batch_size:
                process_batch(model, gpu_lock, device, batch_frames, batch_indices, subject_id, type_str, view, curr_rgb_dir, curr_mask_dir, frame_w, frame_h, args)
                batch_frames = []
                batch_indices = []
            
            frame_idx += 1
        
        if len(batch_frames) > 0:
            process_batch(model, gpu_lock, device, batch_frames, batch_indices, subject_id, type_str, view, curr_rgb_dir, curr_mask_dir, frame_w, frame_h, args)
            
        cap.release()
    except Exception as e:
        # 只打印错误，不打印进度
        print(f"Error processing {video_file}: {e}", flush=True)

# ================== GPU 工作进程 ==================
def gpu_worker(gpu_id, video_subset, args, rgb_root, mask_root):
    """
    每个 GPU 进程的工作函数
    """
    device = f"cuda:{gpu_id}"
    print(f"🚀 [GPU {gpu_id}] 启动! 准备加载模型到 {device}...", flush=True)
    
    try:
        model = YOLO(args.model)
        model.to(device)
        # 预热
        model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device)
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] 模型加载失败: {e}", flush=True)
        return

    gpu_lock = threading.Lock()
    total = len(video_subset)
    print(f"▶ [GPU {gpu_id}] 开始处理 {total} 个视频 (Threads={args.workers})...", flush=True)

    start_time = time.time()
    
    # 启动线程池
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_one_video, f, model, gpu_lock, device, rgb_root, mask_root, args) 
            for f in video_subset
        ]
        
        # 【修改】移除了中间的循环打印，只等待全部完成
        concurrent.futures.wait(futures)
            
    print(f"✅ [GPU {gpu_id}] 任务完成! 总耗时: {time.time() - start_time:.1f}s", flush=True)

# ================== 主函数 ==================
def main():
    args = get_args()
    
    gpu_list = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_list)
    print(f">>> 检测到 {num_gpus} 张显卡: {gpu_list}")
    print(f">>> 每个 GPU 将开启 {args.workers} 个 CPU 线程")
    
    rgb_root = os.path.join(args.output_path, 'RGB')
    mask_root = os.path.join(args.output_path, 'Mask')
    os.makedirs(rgb_root, exist_ok=True)
    os.makedirs(mask_root, exist_ok=True)

    if not os.path.exists(args.input_path):
        print(f"❌ 错误：输入路径不存在: {args.input_path}")
        return

    all_videos = [f for f in os.listdir(args.input_path) if f.endswith(('.avi', '.mp4'))]
    if not all_videos:
        print("❌ 未找到视频文件")
        return
    
    total_videos = len(all_videos)
    print(f">>> 总视频数: {total_videos}")

    chunk_size = math.ceil(total_videos / num_gpus)
    processes = []

    mp.set_start_method('spawn', force=True)

    for i, gpu_id in enumerate(gpu_list):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_videos)
        video_subset = all_videos[start_idx:end_idx]
        
        if not video_subset:
            continue
        
        p = mp.Process(
            target=gpu_worker, 
            args=(gpu_id, video_subset, args, rgb_root, mask_root)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("\n>>> 🎉 所有 GPU 任务全部完成！")

if __name__ == '__main__':
    main()