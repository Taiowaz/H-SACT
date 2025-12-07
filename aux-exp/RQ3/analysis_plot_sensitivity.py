import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re

# ================= 配置区 =================
DATASETS = ["thgl-software-subset", "thgl-github-subset"]  # 两列数据集
FEATURES = [
    {
        "name": "rgfm_embed_dim",
        "nums": [8, 16, 32, 64, 128],
        "label": "Riemannian Embedding Dimension ($d$)",
        "xscale": "log",
        "base": 2
    },
    {
        "name": "window_size",
        "nums": [2, 5, 10, 25, 50],
        "label": "Window Size ($w$)",
        "xscale": "linear", # 或者 log
        "base": 10
    },
    {
        "name": "structure_time_gap",
        "nums": [500, 1000, 2000, 4000, 8000],
        "label": "Structure Time Gap ($q$)",
        "xscale": "linear", # 或者 log
        "base": 10
    }
]

BASE_DIR = "exper"

# 如果有 STHN 的基准线，可以在这里配置字典，格式: {"dataset_name": score}
STHN_BASELINE_SCORES = {
    "thgl-software-subset": 0.0, 
    "thgl-forum-subset": 0.0
}
# ========================================

def extract_metric(file_path):
    """从 results.json 中提取 MRR, AUC, AP"""
    if not os.path.exists(file_path):
        # print(f"Warning: File not found {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        def parse_val(val):
            if val is None: return 0.0
            if isinstance(val, (int, float)): return float(val)
            match = re.search(r"(\d+\.\d+)", str(val))
            return float(match.group(1)) if match else 0.0

        mrr = parse_val(data.get("test mrr") or data.get("MRR"))
        auc = parse_val(data.get("test auroc") or data.get("test auc") or data.get("AUC"))
        ap = parse_val(data.get("test auprc") or data.get("test ap") or data.get("AP"))
        return {"test mrr": mrr, "test auroc": auc, "test auprc": ap}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # 创建 3行 2列 的画布
    # 增加顶部边距 (top=0.85) 为标题和图例留出更多空间
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    
    # 用于收集图例句柄
    legend_dict = {}

    # 遍历每一行（特征）
    for row_idx, feature_cfg in enumerate(FEATURES):
        feature_name = feature_cfg["name"]
        nums = feature_cfg["nums"]
        # xlabel = feature_cfg["label"]
        
        # 遍历每一列（数据集）
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            
            res_mrr, res_auc, res_ap, valid_nums = [], [], [], []
            
            print(f"📊 处理 {dataset} - {feature_name} ...")
            
            for num in nums:
                exp_name = f"sensitivity_{feature_name}_{num}"
                json_path = os.path.join(BASE_DIR, exp_name, dataset, "result", f"{dataset}_results.json")
                
                score_dict = extract_metric(json_path)
                if score_dict:
                    res_mrr.append(score_dict["test mrr"])
                    res_auc.append(score_dict["test auroc"])
                    res_ap.append(score_dict["test auprc"])
                    valid_nums.append(num)

            if not valid_nums:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue

            # 使用索引作为 x 坐标
            x_indices = [nums.index(n) for n in valid_nums]

            # 绘图
            l1, = ax.plot(x_indices, res_mrr, 'o-', color='#FF8C00', linewidth=2, label='MRR', markersize=6)
            l2, = ax.plot(x_indices, res_auc, 's--', color='#1E90FF', linewidth=2, label='AUC', markersize=6)   
            l3, = ax.plot(x_indices, res_ap, 'd-.', color='#32CD32', linewidth=2, label='AP', markersize=6)

            # 收集图例
            if 'MRR' not in legend_dict: legend_dict['MRR'] = l1
            if 'AUC' not in legend_dict: legend_dict['AUC'] = l2
            if 'AP' not in legend_dict:  legend_dict['AP'] = l3

            # STHN Baseline
            baseline = STHN_BASELINE_SCORES.get(dataset, 0.0)
            if baseline > 0:
                l_base = ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, label='STHN')
                if 'STHN' not in legend_dict: legend_dict['STHN'] = l_base

            # 设置坐标轴
            ax.set_xticks(range(len(nums)))
            ax.set_xticklabels(nums)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # === 修改: 移除 ax.set_title，改用全局文本 ===
            # if row_idx == 0:
            #     ax.set_title(dataset, fontsize=14, fontweight='bold')
            
            # 左侧 Y 轴标签
            if col_idx == 0:
                ax.set_ylabel(feature_cfg["label"], fontsize=12, fontweight='bold')
            
            # 底部 X 轴标签
            # if row_idx == 2:
            #     ax.set_xlabel(xlabel, fontsize=12)

    # === 布局调整 ===
    # 先调用 tight_layout 自动调整子图间距
    plt.tight_layout()
    # 调整顶部边距，留出空间给标题和图例 (单行布局，不需要太高，0.92即可)
    plt.subplots_adjust(top=0.94)

    # === 1. 计算位置 ===
    # 获取两列子图的中心位置
    pos_left = axes[0, 0].get_position()
    pos_right = axes[0, 1].get_position()
    
    center_left = (pos_left.x0 + pos_left.x1) / 2
    center_right = (pos_right.x0 + pos_right.x1) / 2
    
    # 【关键修改】计算两个标题的几何中心点，确保图例严格居中于两个标题之间
    legend_center_x = (center_left + center_right) / 2
    
    # 设定统一的 Y 坐标，让它们在同一行
    header_y = 0.96

    # === 2. 放置数据集标题 (左右两侧) ===
    # 稍微减小字体以防重叠
    fig.text(center_left, header_y, DATASETS[0], ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(center_right, header_y, DATASETS[1], ha='center', va='center', fontsize=14, fontweight='bold')

    # === 3. 放置图例 (居中，与标题同排) ===
    # 将图例放在两列标题的中间
    # 【关键修改】bbox_to_anchor 使用计算出的 legend_center_x 而不是固定的 0.5
    fig.legend(legend_dict.values(), legend_dict.keys(), 
               loc='center', bbox_to_anchor=(legend_center_x, header_y), 
               ncol=len(legend_dict), fontsize=10, 
               handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
               frameon=False)

    save_dir = "aux_exp/RQ3/data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sensitivity_combined_3x2.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 组合图表已保存至: {save_path}")

if __name__ == "__main__":
    main()