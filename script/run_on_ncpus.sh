#!/bin/bash

CPUS_PER_TASK=1
TOTAL_CPUS=0
DRY_RUN=0

usage() {
    echo "Usage: $0 [--cpus-per-task N] [--total-cpus N] [--dry-run] <command_file> [command_file ...]"
    exit 1
}

while [[ "$1" =~ ^- ]]; do
    case $1 in
        --cpus-per-task)
            shift
            [[ "$1" =~ ^[0-9]+$ ]] || usage
            CPUS_PER_TASK="$1"
            ;;
        --total-cpus)
            shift
            [[ "$1" =~ ^[0-9]+$ ]] || usage
            TOTAL_CPUS="$1"
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            usage
            ;;
    esac
    shift
done

CMD_FILES=("$@")
[[ "${#CMD_FILES[@]}" -eq 0 ]] && usage

if [ "$TOTAL_CPUS" -eq 0 ]; then
    if [ -n "$SLURM_CPUS_ON_NODE" ]; then
        TOTAL_CPUS="$SLURM_CPUS_ON_NODE"
    elif [ -n "$SLURM_JOB_CPUS_PER_NODE" ]; then
        TOTAL_CPUS="${SLURM_JOB_CPUS_PER_NODE%%(*}"
        TOTAL_CPUS="${TOTAL_CPUS%%,*}"
    else
        echo "Error: Could not detect total CPUs. Pass --total-cpus explicitly."
        exit 1
    fi
fi

MAX_PARALLEL=$(( TOTAL_CPUS / CPUS_PER_TASK ))
if [ "$MAX_PARALLEL" -lt 1 ]; then
    echo "Error: Allocation too small for cpus-per-task=$CPUS_PER_TASK"
    exit 1
fi

declare -a COMMANDS=()
declare -a PIDS=()
declare -a TASK_IDS=()
declare -A STATUS=()

for cmd_file in "${CMD_FILES[@]}"; do
    if [[ ! -f "$cmd_file" ]]; then
        echo "Error: Command file not found: $cmd_file"
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        COMMANDS+=("$line")
    done < "$cmd_file"
done

TOTAL_LINES=${#COMMANDS[@]}
CURRENT_LINE=0

mkdir -p logs

echo "==========================================="
echo "     CPU Task Launcher"
echo "==========================================="
echo "CPUs per task : $CPUS_PER_TASK"
echo "Total CPUs    : $TOTAL_CPUS"
echo "Max parallel  : $MAX_PARALLEL"
echo "==========================================="

launch_task() {
    local line_num=$1
    local cmd="${COMMANDS[$((line_num - 1))]}"
    [[ -z "$cmd" ]] && return

    local task_id
    task_id=$(printf "%02d" "$line_num")
    local log_file="logs/task_${task_id}.log"

    echo "[Task $task_id] -> $cmd"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  (Dry run: skipping srun)"
        return
    fi

    srun --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" --exclusive --cpu-bind=cores \
        bash -c "$cmd" > "$log_file" 2>&1 < /dev/null &

    PIDS+=("$!")
    TASK_IDS+=("$task_id")
    sleep 1
}

while : ; do
    while (( CURRENT_LINE < TOTAL_LINES && ${#PIDS[@]} < MAX_PARALLEL )); do
        ((CURRENT_LINE++))
        launch_task "$CURRENT_LINE"
    done

    if (( CURRENT_LINE >= TOTAL_LINES && ${#PIDS[@]} == 0 )); then
        break
    fi

    if [ "${#PIDS[@]}" -gt 0 ]; then
        wait -n 2>/dev/null
    fi

    active_pids=()
    active_tids=()
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[i]}"
        tid="${TASK_IDS[i]}"
        if kill -0 "$pid" 2>/dev/null; then
            active_pids+=("$pid")
            active_tids+=("$tid")
        else
            wait "$pid"
            rc=$?
            if [ $rc -eq 0 ]; then
                STATUS[$tid]="Success"
            else
                STATUS[$tid]="Failed (exit $rc)"
            fi
        fi
    done
    PIDS=("${active_pids[@]}")
    TASK_IDS=("${active_tids[@]}")
done

echo
echo "==================== Summary ===================="
for ((i=1; i<=TOTAL_LINES; i++)); do
    tid=$(printf "%02d" "$i")
    status="${STATUS[$tid]}"
    [[ -z "$status" ]] && status="Unknown"
    echo "Task $tid: $status"
done
echo "================================================="
