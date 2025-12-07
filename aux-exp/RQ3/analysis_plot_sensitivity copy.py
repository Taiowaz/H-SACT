import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re

# ================= 配置区 =================
DATASET = "thgl-software-subset"
feature="rgfm_embed_dim"
NUMS = [8, 16, 32, 64,128]
# feature="window_size"
# NUMS=[2,5,10,25,50] 
# feature="structure_time_gap"
# NUMS=[500,1000,2000,4000,8000]
BASE_DIR = "exper"
METRIC = "test mrr" 

# 🟢 在这里填入你之前跑出的 STHN (abl_sthn) 的结果
# 如果你还没跑完或者找不到，可以先填个 0.72 这种假数据看看效果
STHN_BASELINE_SCORE = 0.0 # 举例：假设 STHN 的结果是 0.742
# ========================================

def extract_metric(file_path):
    """从 results.json 中提取 MRR, AUC, AP"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        def parse_val(val):
            if val is None: return 0.0
            if isinstance(val, (int, float)): return float(val)
            # 处理可能的字符串格式
            match = re.search(r"(\d+\.\d+)", str(val))
            return float(match.group(1)) if match else 0.0

        # 提取三个指标
        mrr = parse_val(data.get("test mrr") or data.get("MRR"))
        auc = parse_val(data.get("test auroc") or data.get("test auc") or data.get("AUC"))
        ap = parse_val(data.get("test auprc") or data.get("test ap") or data.get("AP"))
        metrics = {"test mrr": mrr, "test auroc": auc, "test auprc": ap}
        print(f"  提取指标: MRR={mrr}, AUC={auc}, AP={ap}")
        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    res_mrr = []
    res_auc = []
    res_ap = []
    valid_nums = []
    
    print(f"📊 正在收集 {DATASET} {feature} 的 GeoSTHN 数据...")
    for num in NUMS:
        exp_name = f"sensitivity_{feature}_{num}"
        json_path = os.path.join(BASE_DIR, exp_name, DATASET, "result", f"{DATASET}_results.json")
        score_dict = extract_metric(json_path)
        res_mrr.append(score_dict["test mrr"])
        res_auc.append(score_dict["test auroc"])
        res_ap.append(score_dict["test auprc"])
        valid_nums.append(num)

    plt.figure(figsize=(8, 6))
    
    plt.plot(valid_nums, res_mrr, 'o-', color='#FF8C00', linewidth=3, label='MRR', markersize=9)
    plt.plot(valid_nums, res_auc, 's--', color='#1E90FF', linewidth=3, label='AUC', markersize=9)   
    plt.plot(valid_nums, res_ap, 'd-.', color='#32CD32', linewidth=3, label='AP', markersize=9)

    if STHN_BASELINE_SCORE > 0:
        plt.axhline(y=STHN_BASELINE_SCORE, color='gray', linestyle='--', linewidth=2, label=f'STHN (Best: {STHN_BASELINE_SCORE:.4f})')

    # plt.title(f"Parameter Sensitivity: Embedding Dimension ($d$)", fontsize=16)
    # plt.xlabel("Riemannian Embedding Dimension", fontsize=14)
    # plt.ylabel(f"Performance", fontsize=14)
    
    # 设置对数坐标轴，让 4, 8, 16, 32 等距分布
    plt.xscale('log', base=2) 
    plt.xticks(NUMS, NUMS, fontsize=12) # 强制显示刻度
    plt.yticks(fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, loc='lower right') # 图例放右下角
    plt.tight_layout()
    
    save_path = f"aux_exp/RQ3/data/sensitivity_{feature}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已保存至: {save_path}")


if __name__ == "__main__":
    main()