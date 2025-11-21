import torch

def get_collate_fn(processor, mode='train'):
    """
    返回对应模式的 collate_fn
    自动处理 Train(单帧) 和 Test(多帧) 的维度差异
    以及 List[PIL] 和 Tensor 的兼容性
    """
    def collate_fn(batch):
        # 1. 过滤 None (健壮性)
        batch = [x for x in batch if x is not None]
        if len(batch) == 0: return None

        # 2. 提取数据
        # ref_raw/tar_raw 可能是:
        #   - Tensor (已处理)
        #   - List[PIL] (未处理的多帧)
        #   - PIL (未处理的单帧)
        ref_raw = [item['ref_imgs'] for item in batch]
        tar_raw = [item['tar_imgs'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # === 3. 处理图片 (核心修复部分) ===
        
        # 情况 A: Dataset 返回的是 Tensor (最高效，假设已处理好)
        if torch.is_tensor(ref_raw[0]):
            ref_stack = torch.stack(ref_raw)
            tar_stack = torch.stack(tar_raw)
            
        # 情况 B: Dataset 返回的是 List[PIL] (测试模式多帧，未处理)
        elif isinstance(ref_raw[0], list):
            batch_size = len(batch)
            frames_num = len(ref_raw[0])
            
            # 展平: [[8帧], [8帧]] -> [16帧]
            flat_ref = [img for seq in ref_raw for img in seq]
            flat_tar = [img for seq in tar_raw for img in seq]
            
            # 统一处理
            inputs = processor(text=texts, images=flat_ref + flat_tar, return_tensors="pt", padding=True, truncation=True)
            
            # 拆分并重塑
            total_pixels = inputs['pixel_values']
            mid_point = len(flat_ref)
            
            ref_pix_flat = total_pixels[:mid_point]
            tar_pix_flat = total_pixels[mid_point:]
            
            # Reshape: [B*T, C, H, W] -> [B, T, C, H, W]
            ref_stack = ref_pix_flat.view(batch_size, frames_num, 3, 224, 224)
            tar_stack = tar_pix_flat.view(batch_size, frames_num, 3, 224, 224)
            
        # 情况 C: Dataset 返回的是单张 PIL (训练模式单帧，未处理)
        else:
            inputs = processor(text=texts, images=ref_raw + tar_raw, return_tensors="pt", padding=True, truncation=True)
            ref_stack, tar_stack = inputs['pixel_values'].chunk(2)

        # === 4. 处理文本 ===
        # 如果 inputs 已经在上面处理图片时顺便生成了 input_ids (情况 B 和 C)
        if 'input_ids' in locals().get('inputs', {}):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        # 如果 Dataset 已经做好了 Tokenization (情况 A)
        elif 'input_ids' in batch[0]:
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
        # 兜底：如果只有图片处理了，文本还没处理
        else:
            text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']

        # === 5. 返回结果 ===
        if mode == 'train':
            return ref_stack, tar_stack, input_ids, attention_mask
        else:
            # 测试模式需要元数据
            tasks = [item['task'] for item in batch]
            meta = []
            for item in batch:
                meta.append({
                    'sid': item['sid'],
                    'cond': item['cond'],
                    'view': item['view']
                })
            return ref_stack, tar_stack, input_ids, attention_mask, tasks, meta

    return collate_fn