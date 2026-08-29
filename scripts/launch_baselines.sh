#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" && -x /root/miniconda3/bin/python ]]; then
    python_bin=/root/miniconda3/bin/python
elif [[ -z "$python_bin" ]]; then
    python_bin="$(command -v python3 || command -v python)"
fi

log_dir="$project_root/artifacts/launcher_logs"
pid_file="$project_root/artifacts/baseline_launcher.pid"
mkdir -p "$log_dir"

if [[ -f "$pid_file" ]]; then
    previous_pid="$(<"$pid_file")"
    if kill -0 "$previous_pid" 2>/dev/null; then
        echo "Baseline launcher is already running with PID $previous_pid"
        exit 1
    fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$log_dir/baselines_${timestamp}.log"
nohup "$python_bin" scripts/run_baselines.py >"$log_file" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$pid_file"

echo "Baseline experiments started with PID $launcher_pid"
echo "Log: $log_file"
echo "Monitor: tail -f $log_file"
