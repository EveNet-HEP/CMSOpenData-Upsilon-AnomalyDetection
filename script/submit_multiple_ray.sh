#!/bin/bash
# Usage ./submit_multiple_ray.sh -p 6 commands_file1.txt commands_file2.txt

# Colors
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[1;34m'
CYN='\033[1;36m'
RST='\033[0m'

# Fancy Banner
echo -e "${CYN}"
echo "╔═══════════════════════════════════════════════╗"
echo "║       🚀 Ray Job Submitter with Throttle      ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${RST}"

# Defaults
max_parallel=4
COMMANDS_FILES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--parallel)
      max_parallel="$2"
      shift 2
      ;;
    *)
      COMMANDS_FILES+=("$1")
      shift
      ;;
  esac
done

# Validate
if [[ ${#COMMANDS_FILES[@]} -eq 0 ]]; then
  echo -e "${RED}❌ Usage: $0 [-p max_parallel] <commands_file1> [commands_file2 ...]${RST}"
  exit 1
fi

# Get Ray address
head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi
port=6379
RAY_ADDRESS=$head_node_ip:$port

echo -e "${BLU}📡 Using Ray address: ${GRN}$RAY_ADDRESS${RST}"
echo -e "${BLU}🔁 Max parallel jobs: ${GRN}$max_parallel${RST}"
echo ""

pids=()

# Loop over files
for COMMANDS_FILE in "${COMMANDS_FILES[@]}"; do
  echo -e "${CYN}📄 Processing file: ${YLW}$COMMANDS_FILE${RST}"
  echo "--------------------------------------------"

  while IFS= read -r cmd || [[ -n "$cmd" ]]; do
    [[ -z "$cmd" ]] && continue

    echo -e "${GRN}➡️  Submitting:${RST} $cmd"
    ray job submit --address="$RAY_ADDRESS" -- bash -c "$cmd" &
    pids+=($!)
    

    sleep 3

    while (( $(jobs -r | wc -l) >= max_parallel )); do
      echo -ne "${YLW}⏳ Waiting for available slot...${RST}\r"
      sleep 60
    done
  done < "$COMMANDS_FILE"
  echo ""
done

# Wait for all jobs
echo -e "${CYN}🧘 Waiting for all background jobs to finish...${RST}"
wait

echo -e "${GRN}✅ All jobs submitted and completed.${RST}"

