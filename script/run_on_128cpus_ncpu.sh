#!/bin/bash

echo "==========================================="
echo "     🚀 NERSC CPU-only Task Launcher        "
echo "    Sliding window with configurable tasks "
echo "==========================================="

# Default CPUs per task
CPUS_PER_TASK=1

usage() {
    echo "Usage: $0 [--cpus-per-task N] <command_file>"
    exit 1
}

# Parse optional args
while [[ "$1" =~ ^- ]]; do
    case $1 in
        --cpus-per-task)
            shift
            [[ "$1" =~ ^[0-9]+$ ]] || usage
            CPUS_PER_TASK="$1"
            ;;
        *)
            usage
            ;;
    esac
    shift
done

CMD_FILE="$1"
if [ -z "$CMD_FILE" ] || [ ! -f "$CMD_FILE" ]; then
    usage
fi

mkdir -p logs

# You can set total CPUs available on the node here:
TOTAL_CPUS=128  # Change as per your system

# Calculate max parallel tasks allowed by CPUs
MAX_PARALLEL=$(( TOTAL_CPUS / CPUS_PER_TASK ))

declare -a PIDS=()
declare -a TASK_IDS=()
declare -A STATUS=()

mapfile -t COMMANDS < "$CMD_FILE"
TOTAL_LINES=${#COMMANDS[@]}
CURRENT_LINE=0

launch_task() {
    local line_num=$1
    local cmd="${COMMANDS[$((line_num - 1))]}"
    [[ -z "$cmd" ]] && return

    local task_id
    task_id=$(printf "%02d" "$line_num")

    echo "[Task $task_id] → $cmd (CPUs: $CPUS_PER_TASK)"
    srun --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --exclusive --cpu-bind=cores \
         bash -c "echo '[Task $task_id] STARTING'; $cmd; echo '[Task $task_id] DONE'" \
         > "logs/task_$task_id.log" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    TASK_IDS+=("$task_id")
    sleep 1
}

while : ; do
    while (( CURRENT_LINE < TOTAL_LINES && ${#PIDS[@]} < MAX_PARALLEL )); do
        ((CURRENT_LINE++))
        launch_task "$CURRENT_LINE"
    done

    if [ "${#PIDS[@]}" -eq 0 ]; then
        break
    fi

    wait -n
    for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            wait "${PIDS[i]}"
            exit_code=$?

            task_id="${TASK_IDS[i]}"
            if [ $exit_code -eq 0 ]; then
                STATUS[$task_id]="✅ Success"
            else
                STATUS[$task_id]="❌ Failed (exit $exit_code)"
            fi

            unset 'PIDS[i]'
            unset 'TASK_IDS[i]'
            PIDS=("${PIDS[@]}")
            TASK_IDS=("${TASK_IDS[@]}")
            break
        fi
    done
done

echo
echo "==================== Summary ===================="
for ((i=1; i<=TOTAL_LINES; i++)); do
    tid=$(printf "%02d" "$i")
    status="${STATUS[$tid]}"
    [[ -z "$status" ]] && status="❓ Unknown"
    echo "Task $tid: $status"
done
echo "================================================="
echo "✅ All tasks processed. Logs are in logs/ directory."

