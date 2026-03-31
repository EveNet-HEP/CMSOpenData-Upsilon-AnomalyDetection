#!/bin/bash

# ===============================================
#    🚀 NERSC Launcher V4 (Debug & Robust)
# ===============================================

# --- Defaults ---
GPUS_PER_TASK=8       
GPUS_PER_NODE=4       
NTASKS_PARAM=""       
TOTAL_NODES_ALLOC=0   
DRY_RUN=0             
TASKS_PER_NODE=1  # Default to 1 task per node (standard behavior)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

usage() {
    echo -e "${RED}Usage: $0 [options] <command_file> [command_file ...]${NC}"
    echo "Options:"
    echo "  --gpus-per-task N   (Default: $GPUS_PER_TASK)"
    echo "  --gpus-per-node N   (Default: $GPUS_PER_NODE)"
    echo "  --ntasks N          (Critical! Set '1' for torchrun/deepspeed. Default: Same as GPU count)"
    echo "  --total-nodes N     (Auto-detect if empty)"
    echo "  -p, --tasks-per-node N"
    echo "  --dry-run           (Print what would happen without running)"
    exit 1
}

# --- Arguments ---
while [[ "$1" =~ ^- ]]; do
    case $1 in
        --gpus-per-task) shift; GPUS_PER_TASK="$1" ;;
        --gpus-per-node) shift; GPUS_PER_NODE="$1" ;;
        --ntasks)        shift; NTASKS_PARAM="$1" ;;
        --total-nodes)   shift; TOTAL_NODES_ALLOC="$1" ;;
        --dry-run)       DRY_RUN=1 ;;
        --tasks-per-node|-p) shift; TASKS_PER_NODE="$1" ;;
        *) usage ;;
    esac
    shift
done

CMD_FILES=("$@")
[[ "${#CMD_FILES[@]}" -eq 0 ]] && usage

# --- Logic Setup ---
# Default logic: If using MPI, tasks = GPUs. If torchrun, user usually wants tasks=1.
if [ -z "$NTASKS_PARAM" ]; then
    NTASKS_PARAM="$GPUS_PER_TASK"
    MODE_MSG="MPI Mode (One task per GPU)"
else
    MODE_MSG="Wrapper Mode (Explicit tasks: $NTASKS_PARAM)"
fi

if [ "$TOTAL_NODES_ALLOC" -eq 0 ]; then
    if [ -n "$SLURM_NNODES" ]; then
        TOTAL_NODES_ALLOC="$SLURM_NNODES"
    else
        echo -e "${RED}Error: Not in SLURM. Bind --total-nodes manually.${NC}"
        exit 1
    fi
fi

# Calculate Nodes Needed
# Formula: ceil(GPUS / PER_NODE)
NODES_PER_TASK=$(( (GPUS_PER_TASK + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
MAX_PARALLEL=$(( (TOTAL_NODES_ALLOC * TASKS_PER_NODE) / NODES_PER_TASK ))
if [ "$MAX_PARALLEL" -lt 1 ]; then
    echo -e "${RED}❌ Error: Allocation too small.${NC}"
    exit 1
fi

echo "========================================================="
echo -e "⚙️  Config: ${CYAN}${GPUS_PER_TASK} GPUs${NC} ($NODES_PER_TASK Nodes) per command"
echo -e "📦 Total Pool: ${CYAN}${TOTAL_NODES_ALLOC} Nodes${NC}"
echo -e "🚀 Max Concurrent Jobs: ${GREEN}${MAX_PARALLEL}${NC}"
echo -e "ℹ️  Slurm Ntasks: ${YELLOW}${NTASKS_PARAM}${NC} ($MODE_MSG)"
echo "========================================================="

if [ "$MAX_PARALLEL" -lt 1 ]; then
    echo -e "${RED}❌ Error: Allocation too small. Need $NODES_PER_TASK nodes per task.${NC}"
    exit 1
fi

declare -a PIDS=()
declare -a TASK_IDS=()
declare -a COMMANDS=()

for cmd_file in "${CMD_FILES[@]}"; do
    if [[ ! -f "$cmd_file" ]]; then
        echo -e "${RED}Error: Command file not found: $cmd_file${NC}"
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        COMMANDS+=("$line")
    done < "$cmd_file"
done

TOTAL_LINES=${#COMMANDS[@]}
CURRENT_LINE=0

mkdir -p logs

# --- The Loop ---
while : ; do
    # 1. Reaper: Clean up dead processes from the PID list
    active_pids=()
    active_tids=()
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[i]}"
        tid="${TASK_IDS[i]}"
        if kill -0 "$pid" 2>/dev/null; then
            active_pids+=("$pid")
            active_tids+=("$tid")
        else
            # Collect exit code
            wait "$pid"
            rc=$?
            if [ $rc -eq 0 ]; then
                echo -e "[Task $tid] ${GREEN}✅ Finished successfully${NC}"
            else
                echo -e "[Task $tid] ${RED}❌ Failed (Exit Code: $rc) - Check logs/task_$tid.log${NC}"
            fi
        fi
    done
    PIDS=("${active_pids[@]}")
    TASK_IDS=("${active_tids[@]}")

    # 2. Launcher: Launch if we have room
    while (( CURRENT_LINE < TOTAL_LINES && ${#PIDS[@]} < MAX_PARALLEL )); do
        ((CURRENT_LINE++))
        cmd="${COMMANDS[$((CURRENT_LINE - 1))]}"
        [[ -z "$cmd" ]] && continue
        
        tid=$(printf "%02d" "$CURRENT_LINE")
        log_file="logs/task_$tid.log"
 
        rand_port=$(shuf -i 20000-60000 -n 1)
        echo -e "[Task $tid] ${YELLOW}Executing:${NC} $cmd (port: $rand_port)"
        
        if [ "$DRY_RUN" -eq 1 ]; then
             echo "   (Dry Run: Skipping srun)"
        else
             # CRITICAL FIXES:
             # 1. < /dev/null : Prevents stdin hijacking
             # 2. sleep 2     : Prevents Slurm scheduler race condition
             
             srun --exclusive \
                  --nodes="$NODES_PER_TASK" \
                  --gpus="$GPUS_PER_TASK" \
                  --ntasks="$NTASKS_PARAM" \
                  --cpus-per-task=10 \
                  --export=ALL,MASTER_PORT="$rand_port" \
                  bash -c "$cmd" \
                  > "$log_file" 2>&1 < /dev/null &
             
             pid=$!
             PIDS+=("$pid")
             TASK_IDS+=("$tid")
             
             echo -e "   → Started PID $pid (Log: $log_file)"
             sleep 2 
        fi
    done

    # 3. Exit Condition
    if (( CURRENT_LINE >= TOTAL_LINES && ${#PIDS[@]} == 0 )); then
        break
    fi

    # 4. Wait smart (Don't busy loop)
    if [ "${#PIDS[@]}" -gt 0 ]; then
        # wait -n is efficient, returns as soon as ONE finishes
        wait -n 2>/dev/null
    fi
done

echo -e "${GREEN}✅ All tasks processed.${NC}"
