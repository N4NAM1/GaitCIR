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
        # 1. 批次数据健壮性检查：过滤掉任何为 None 的样本
        batch = [x for x in batch if x is not None]
        if len(batch) == 0: return None

        # 2. 从批次中提取原始数据
        # ref_raw 和 tar_raw 的数据类型取决于 Dataset 的输出。
        ref_raw = [item['ref_imgs'] for item in batch]
        tar_raw = [item['tar_imgs'] for item in batch]
        texts = [item['text'] for item in batch] # 正向指令 (T_fwd)，用于 L_cir

        # 提取逆向指令 (T_inv)，用于计算循环一致性损失 (L_cycle)
        # 假设 Dataset 中对应的字段名为 'text_inv'
        text_invs = [item['text_inv'] for item in batch]
        
        # 初始化 inputs 字典，用于存储 processor 的输出，方便后续判断
        inputs = {} 
        
        # === 3. 处理图片：统一转换为 Tensor 堆栈 ===
        
        # 情况 A: Dataset 直接返回 Tensor (最高效)，表示图像已在 Dataset 阶段预处理完成
        if torch.is_tensor(ref_raw[0]):
            ref_stack = torch.stack(ref_raw)
            tar_stack = torch.stack(tar_raw)
            
        # 情况 B: Dataset 返回 List[PIL.Image] (多帧/序列数据，常见于测试阶段)
        elif isinstance(ref_raw[0], list):
            batch_size = len(batch)
            frames_num = len(ref_raw[0])
            
            # 展平图像列表：将 [B, T] 结构压平为 [B*T] 的一维列表
            # 以便 Processor 可以一次性高效处理所有帧
            flat_ref = [img for seq in ref_raw for img in seq]
            flat_tar = [img for seq in tar_raw for img in seq]
            
            # 使用 Processor 统一处理图像和文本。
            # 图像预处理、归一化，文本 Tokenization 在此同步完成。
            inputs = processor(text=texts, images=flat_ref + flat_tar, return_tensors="pt", padding=True, truncation=True)
            
            # 从像素值中分离出 Ref 和 Tar 的数据
            total_pixels = inputs['pixel_values']
            mid_point = len(flat_ref) # 分界点是 Ref 帧的总数
            
            ref_pix_flat = total_pixels[:mid_point]
            tar_pix_flat = total_pixels[mid_point:]
            
            # Reshape：将处理后的张量从 [B*T, C, H, W] 重新塑形为 [B, T, C, H, W]
            ref_stack = ref_pix_flat.view(batch_size, frames_num, 3, 224, 224)
            tar_stack = tar_pix_flat.view(batch_size, frames_num, 3, 224, 224)
            
        # 情况 C: Dataset 返回单张 PIL.Image (单帧数据，常见于训练阶段)
        else:
            # 统一处理 Ref 和 Tar 图像，并将 texts 同时 Tokenize
            inputs = processor(text=texts, images=ref_raw + tar_raw, return_tensors="pt", padding=True, truncation=True)
            # 像素值被 processor 自动按 Ref/Tar 顺序拼接，这里使用 chunk 拆分
            ref_stack, tar_stack = inputs['pixel_values'].chunk(2)

        # === 4. 处理文本特征 Tokenization ===
        
        # 4.1 处理正向指令 (texts)
        # 优先使用第 3 步处理图像时顺带生成的 input_ids (情况 B 和 C)
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        # 其次检查 Dataset 是否已预先 Tokenize (情况 A)
        elif 'input_ids' in batch[0]:
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
        # 最后作为兜底方案：单独使用 processor 处理文本
        else:
            text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']

        # 4.2 处理逆向指令 (text_invs)
        # 逆向指令与正向指令内容不同，需要单独 Tokenization，用于 L_cycle 损失
        # 优先检查 Dataset 是否已预处理（假设字段名为 'input_ids_inv'）
        if 'input_ids_inv' in batch[0]:
            input_ids_inv = torch.stack([item['input_ids_inv'] for item in batch])
            attention_mask_inv = torch.stack([item['attention_mask_inv'] for item in batch])
        else:
            # 如果未预处理，则使用 processor 现场进行 Tokenization
            text_inv_inputs = processor(text=text_invs, padding=True, truncation=True, return_tensors="pt")
            input_ids_inv = text_inv_inputs['input_ids']
            attention_mask_inv = text_inv_inputs['attention_mask']

        # === 5. 返回结果：根据模式返回所需数据 ===
        if mode == 'train':
            # 训练模式：返回图像堆栈、正向文本 (L_cir) 和逆向文本 (L_cycle) 的所有张量
            return ref_stack, tar_stack, input_ids, attention_mask, input_ids_inv, attention_mask_inv
        else:
            # 测试模式：只需要正向文本和元数据 (用于结果评估和定位)
            tasks = [item['task'] for item in batch]
            meta = []
            for item in batch:
                meta.append({
                    'sid': item['sid'], # Subject ID
                    'cond': item['cond'], # Condition 步态条件
                    'view': item['view'] # View 视角
                })
            # 测试模式不返回逆向文本相关的 IDs
            return ref_stack, tar_stack, input_ids, attention_mask, tasks, meta

    return collate_fn