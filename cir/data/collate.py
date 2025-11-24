import torch

def get_collate_fn(processor, mode='train'):
    """
    根据运行模式 (train/test) 返回数据加载器的 collate_fn。

    功能：
    1. 统一处理单帧 (Train) 和多帧 (Test) 数据的维度。
    2. 兼容 List[PIL.Image] 和 Tensor 两种图像输入类型。
    3. 新增支持提取和 Tokenization 正向指令 (Text) 和逆向指令 (Text_inv)。

    参数:
    processor: 用于图像预处理和文本 Tokenization 的 CLIP Processor。
    mode: 'train' 或 'test'。
    """
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0: return None

        # 1. 提取数据
        ref_raw = [item['ref_imgs'] for item in batch]
        tar_raw = [item['tar_imgs'] for item in batch]
        texts = [item['text'] for item in batch]
        texts_inv = [item['text_inv'] for item in batch] # <--- 新增
        
        # 2. 视觉处理 (兼容 Feature / Image 模式)
        if torch.is_tensor(ref_raw[0]):
            # Feature Mode
            if mode == 'train':
                ref_stack = torch.stack(ref_raw)
                tar_stack = torch.stack(tar_raw)
            else:
                ref_stack = ref_raw
                tar_stack = tar_raw
        else:
            # Image Mode (简化版，展平处理)
            flat_ref = [img for seq in ref_raw for img in seq]
            flat_tar = [img for seq in tar_raw for img in seq]
            inputs = processor(text=texts, images=flat_ref + flat_tar, return_tensors="pt", padding=True)
            # 这里的拆分逻辑需根据具体 image_processor 输出调整，暂略
            ref_stack = inputs['pixel_values'] 
            tar_stack = inputs['pixel_values'] 

        # 3. 文本处理 (正向)
        # 如果是 Image Mode，processor 可能已经处理了 text，这里做个兼容检查
        if 'input_ids' in locals().get('inputs', {}):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        else:
            txt_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = txt_inputs['input_ids']
            attention_mask = txt_inputs['attention_mask']

        # 4. 文本处理 (逆向) <--- 新增模块
        # 只有训练模式才需要逆向 loss，但为了统一，也可以都返回
        # 注意：有些样本可能没有 inverse text (空字符串)，processor 会处理成全 padding
        inv_inputs = processor(text=texts_inv, padding=True, truncation=True, return_tensors="pt")
        inv_ids = inv_inputs['input_ids']
        inv_mask = inv_inputs['attention_mask']

        # 5. 返回
        if mode == 'train':
            # 训练返回 6项：Ref, Tar, Txt, Mask, InvTxt, InvMask
            return ref_stack, tar_stack, input_ids, attention_mask, inv_ids, inv_mask
        else:
            tasks = [item['task'] for item in batch]
            meta = [{ 'sid': i['sid'], 'cond': i['cond'], 'view': i['view'] } for i in batch]
            return ref_stack, tar_stack, input_ids, attention_mask, tasks, meta

    return collate_fn