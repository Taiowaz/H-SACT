#!/usr/bin/env bash
set -euo pipefail

run_python="/home/handb/.conda/envs/hsact/bin/python"
run_file="/home/handb/H-SACT/src/main.py"

exper_name="curvature_sensitivity"
dataset="thgl-forum-subset"

mkdir -p /home/handb/H-SACT/run_log/curvature_sensitivity

main_log="/home/handb/H-SACT/run_log/run_${dataset}_${exper_name}.log"

common_args="
  --exper_name ${exper_name}
  --dataset ${dataset}
  --model hetero_sthn
  --use_graph_structure
  --use_cached_subgraph
  --use_riemannian_structure
  --curvature_mode fixed
  --num_run 1
  --num_epoch 5
  --istrain 1
  --use_gpu 0
"

run_one() {
  local kappa="$1"
  local sign_h="$2"
  local sign_s="$3"

  local tag="k${kappa}_h${sign_h}_s${sign_s}"
  local log_file="/home/handb/H-SACT/run_log/curvature_sensitivity/${dataset}_${tag}.log"
  local pid_file="/home/handb/H-SACT/run_log/curvature_sensitivity/${dataset}_${tag}.pid"

  echo "[$(date '+%F %T')] START ${tag}" | tee -a "${main_log}"
  nohup ${run_python} ${run_file} \
    ${common_args} \
    --kappa "${kappa}" \
    --kappa_sign_h "${sign_h}" \
    --kappa_sign_s "${sign_s}" > "${log_file}" 2>&1 &

  echo $! > "${pid_file}"
  echo "[$(date '+%F %T')] SUBMIT ${tag}, PID=$(cat "${pid_file}")" | tee -a "${main_log}"
  echo "LOG: ${log_file}"
}

# =========================================================
# 手动运行：每次只解开一行注释，然后执行本脚本
# =========================================================

# run_one 0.1 -1  1
# run_one 0.5 -1  1
# run_one 1.0 -1  1
# run_one 2.0 -1  1
# run_one 5.0 -1  1

# run_one 0.1  1 -1
# run_one 0.5  1 -1
# run_one 1.0  1 -1
# run_one 2.0  1 -1
run_one 5.0  1 -1

echo "Done. 请确认你已解开一行 run_one 注释。"