export CUDA_LAUNCH_BLOCKING=1
run_python="/home/handb/.conda/envs/geosthn/bin/python"
run_file="/home/handb/GeoSTHN/src/main.py"

exper_name=$(basename "$0" .sh)
common_args="
    --use_graph_structure
    --model hetero_sthn
    --use_cached_subgraph
    --use_riemannian_structure
    --num_epoch 1
    --num_run 1
    --istrain 0
    --use_ali_loss 1
"



dataset="thgl-forum-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-github-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 0 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid


dataset="thgl-myket-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-software-subset"
nohup $run_python $run_file \
    --exper_name $exper_name \
    --dataset $dataset \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid
