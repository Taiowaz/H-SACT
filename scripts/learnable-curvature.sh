#!/usr/bin/env bash
set -euo pipefail

run_python="/home/albin/.conda/envs/hsact/bin/python"
run_file="/home/albin/H-SACT/src/main.py"

mkdir -p run_log

dataset="thgl-forum-subset"
gpu_id=0

common_args="
    --use_onehot_node_feats
    --use_graph_structure
    --model hetero_sthn
    --use_riemannian_structure
"

# 你也可以在这里加统一超参，比如：
# common_args="$common_args --num_run 3 --seed 42"

# -----------------------------
# 1) Fixed curvature baseline
# -----------------------------
exp_fixed="learnable-curvature-fixed"
nohup $run_python $run_file \
    --exper_name "${exp_fixed}" \
    --dataset "${dataset}" \
    $common_args \
    --learnable_curvature 0 \
    --k_h_init 1.0 \
    --k_s_init 1.0 \
    --use_gpu 0 \
    --device ${gpu_id} \
    > "run_log/run_${dataset}_fixed.log" 2>&1 &

echo $! > "run_log/run_${dataset}_fixed.pid"
echo "[Launched] fixed curvature, pid=$(cat run_log/run_${dataset}_fixed.pid)"

# --------------------------------
# 2) Learnable curvature ablation
# --------------------------------
exp_learnable="learnable-curvature-learnable"
nohup $run_python $run_file \
    --exper_name "${exp_learnable}" \
    --dataset "${dataset}" \
    $common_args \
    --learnable_curvature 1 \
    --k_h_init 1.0 \
    --k_s_init 1.0 \
    --curvature_lr_scale 0.1 \
    --curvature_clip_min 1e-3 \
    --curvature_clip_max 1e3 \
    --use_gpu 0 \
    --device ${gpu_id} \
    > "run_log/run_${dataset}_learnable.log" 2>&1 &

echo $! > "run_log/run_${dataset}_learnable.pid"
echo "[Launched] learnable curvature, pid=$(cat run_log/run_${dataset}_learnable.pid)"