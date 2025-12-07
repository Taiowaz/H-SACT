# dataset="forum"
# dataset="myket"
dataset="github"

nohup ./scripts/sensitivity.sh > run_log/${dataset}_sensitivity.log 2>&1 &
PID=$!
echo "$PID" > run_log/${dataset}_sensitivity.pid