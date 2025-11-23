import json
import os

# ================= 配置 =================
INPUT_JSON = './casiab_cir_final_train.json'
OUTPUT_DIR = './' # 保存目录

# CASIA-B 标准分割点
# 训练集: 001 - 074 (共74人)
# 测试集: 075 - 124 (共50人)
TRAIN_ID_CUTOFF = 74 
# =======================================

def split_dataset():
    print(f"正在读取: {INPUT_JSON} ...")
    with open(INPUT_JSON, 'r') as f:
        all_data = json.load(f)
        
    train_set = []
    test_set = []
    
    # 统计 ID 分布
    all_sids = set()
    
    for item in all_data:
        sid_str = item['sid'] # "001", "090"
        sid = int(sid_str)
        all_sids.add(sid)
        
        if sid <= TRAIN_ID_CUTOFF:
            train_set.append(item)
        else:
            test_set.append(item)
            
    # 保存
    train_path = os.path.join(OUTPUT_DIR, 'casiab_cir_train_split.json')
    test_path = os.path.join(OUTPUT_DIR, 'casiab_cir_test_split.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_set, f, indent=4)
        
    with open(test_path, 'w') as f:
        json.dump(test_set, f, indent=4)
        
    print("-" * 30)
    print(f"分割完成！")
    print(f"总 ID 数: {len(all_sids)}")
    print(f"训练集 (ID <= {TRAIN_ID_CUTOFF}): {len(train_set)} 条 -> 保存至 {os.path.basename(train_path)}")
    print(f"测试集 (ID > {TRAIN_ID_CUTOFF}): {len(test_set)} 条 -> 保存至 {os.path.basename(test_path)}")
    print("-" * 30)

if __name__ == '__main__':
    split_dataset()