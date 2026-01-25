#!/bin/bash
# monitor_heartbeat.sh
LOG_FILE="outputs/glm4.7-rust-hybrid-fp4-v3/train.log"

echo "🩺 HEARTBEAT MONITOR ACTIVE"
echo "Waiting for logs..."

while [ ! -f "$LOG_FILE" ]; do sleep 2; done

tail -f "$LOG_FILE" | grep --line-buffered "'loss':" | while read line; do
    LOSS=$(echo "$line" | grep -oP "'loss':\s*\K[\d.]+")
    STEP=$(echo "$line" | grep -oP "'step':\s*\K\d+")
    
    if (( $(echo "$LOSS < 2.0" | bc -l) )); then
        COLOR="\033[1;32m" # Green
    elif (( $(echo "$LOSS < 3.5" | bc -l) )); then
        COLOR="\033[1;36m" # Cyan
    else
        COLOR="\033[1;31m" # Red
    fi
    
    echo -e "${COLOR}[Step $STEP] Loss: $LOSS\033[0m"
done
