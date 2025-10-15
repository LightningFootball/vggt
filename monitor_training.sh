#!/bin/bash
# Training monitor script - Shows CPU/GPU memory and utilization

echo "Training Monitor - Press Ctrl+C to stop"
echo "========================================"
echo ""

while true; do
    clear

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              VGGT Training Monitor                         ║"
    echo "║              $(date '+%Y-%m-%d %H:%M:%S')                           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # CPU Memory
    echo "┌─ CPU Memory ─────────────────────────────────────────────┐"
    free -h | awk 'NR==1 || NR==2 {print "│ " $0}'

    # Calculate memory percentage
    MEM_PERCENT=$(free | awk 'NR==2 {printf "%.1f", $3/$2 * 100}')
    echo "│ Usage: ${MEM_PERCENT}%"

    # Memory warning
    if (( $(echo "$MEM_PERCENT > 80" | bc -l) )); then
        echo "│ ⚠️  WARNING: Memory usage high!"
    elif (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
        echo "│ 🔴 CRITICAL: Memory usage very high!"
    else
        echo "│ ✅ Memory usage normal"
    fi
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    # GPU Memory and Utilization
    echo "┌─ GPU Status ─────────────────────────────────────────────┐"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | \
        awk -F', ' '{printf "│ GPU %s: %s\n│   Memory: %s/%s MB (%.1f%%)\n│   Utilization: %s%%\n│   Temp: %s°C | Power: %sW\n",
            $1, $2, $3, $4, ($3/$4)*100, $5, $6, $7}'
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    # Python processes (training)
    echo "┌─ Training Process ───────────────────────────────────────┐"
    if ps aux | grep -q "[p]ython.*launch.py"; then
        ps aux | grep "[p]ython.*launch.py" | head -1 | \
            awk '{printf "│ PID: %s | CPU: %s%% | RAM: %s%%\n", $2, $3, $4}'

        # Count worker processes
        WORKER_COUNT=$(ps aux | grep -c "[p]ython.*multiprocessing")
        echo "│ DataLoader Workers: $WORKER_COUNT"

        # Check for log file
        if [ -f "logs/lora_kitti360_strategy_b_r16/train.log" ]; then
            LAST_LOG=$(tail -3 logs/lora_kitti360_strategy_b_r16/train.log | grep -E "loss|epoch|iter" | tail -1)
            if [ ! -z "$LAST_LOG" ]; then
                echo "│ Latest: $LAST_LOG" | cut -c1-62
            fi
        fi
    else
        echo "│ ⚠️  No training process detected"
    fi
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    # Recommendations
    if (( $(echo "$MEM_PERCENT > 85" | bc -l) )); then
        echo "💡 Recommendation: Consider reducing num_workers or prefetch_factor"
    fi

    sleep 2
done
