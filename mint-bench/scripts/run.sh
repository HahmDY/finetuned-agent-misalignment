#!/bin/bash
export ALFWORLD_DATA=data/processed/alfworld
export PYTHONPATH=$(pwd):$PYTHONPATH

# For debug
if (($DEBUG_MODE)); then
    RUN_IN_BACKGROUND=0
    N_PARALLEL=1
else
    RUN_IN_BACKGROUND=1
    N_PARALLEL=8
fi

declare -a PIDS=()

trap 'kill_background_processes' INT

function kill_background_processes() {
    echo -e '\nReceived Ctrl+C. Killing background processes...'
    for pid in "${PIDS[@]}"; do
        if ps -p $pid >/dev/null; then
            echo "Killing process $pid"
            kill $pid
        fi
    done
    exit 1
}

function remove_from_array {
    local -n arr=$1
    local value=$2
    local index=0
    for i in "${arr[@]}"; do
        if [ "$i" = "$value" ]; then
            break
        fi
        index=$((index + 1))
    done
    unset 'arr[$index]'
    arr=("${arr[@]}")
}

function run_config_glob() {
    local config_array=("$@")
    local num_configs=${#config_array[@]}
    echo "Total number of configurations: $num_configs"

    if ((num_configs > 0)); then
        echo "Matched configurations:"
        for config in "${config_array[@]}"; do
            echo "- $config"
        done
    else
        echo "No configurations found for the given glob pattern."
    fi

    for config in "${config_array[@]}"; do
        echo "Running $config"
        output_dir=$(jq -r .output_dir "$config")
        mkdir -p "$output_dir"
        echo "=== START DATE: $(date) ===" >>"$output_dir/output.txt"
        if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null; then
            echo "# GIT COMMIT: $(git rev-parse HEAD) ===" >>"$output_dir/output.txt"
        fi

        if (($RUN_IN_BACKGROUND)); then
            python -u mint/main.py --split train --exp_config="$config" --prefix "" >>"$output_dir/output.txt" 2>&1 &
            cur_pid=$!
            echo -e "\n** Started process $cur_pid (run in background). To track progress, run:"
            echo -e "  tail -f $output_dir/output.txt"
            PIDS+=("$cur_pid")

            while ((${#PIDS[@]} >= N_PARALLEL)); do
                for pid in "${PIDS[@]}"; do
                    if ! ps -p "$pid" >/dev/null; then
                        echo "Process $pid finished. Remaining processes: ${PIDS[@]}"
                        remove_from_array PIDS "$pid"
                    fi
                done
                sleep 1
            done
        else
            python -u -m pdb -c continue mint/main.py --split train --exp_config="$config" --prefix "" 2>&1 | tee -a "$output_dir/output.txt"
        fi
    done
}

# ========================

MODEL_NAME=sft_model
FEEDBACK_MODEL=None

run_config_glob configs/alfworld_config.json

# ========================
for pid in "${PIDS[@]}"; do
    wait $pid
done
