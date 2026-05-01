#!/bin/bash
# Comparativa secuencial: FedAvg, FedProx, FedAvgM
set +e

LOG_DIR="logs_comparativa"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MAIN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

export ALMENDROS_DATA_ROOT="${ALMENDROS_DATA_ROOT:-$HOME/TFM/Flower_ejemplo/almendros/almendros_fl/data}"
export ALMENDROS_RESULTS_ROOT="${ALMENDROS_RESULTS_ROOT:-$HOME/TFM/Flower_ejemplo/almendros/almendros_fl/results}"

cleanup_flower() {
    pkill -9 -f flwr 2>/dev/null
    pkill -9 -f flower-superlink 2>/dev/null
    pkill -9 -f ray 2>/dev/null
    rm -f ~/.flwr/local-superlink/state.db 2>/dev/null
    rm -rf /tmp/ray/session_* 2>/dev/null
    sleep 5
}

echo "============================================" | tee -a "$MAIN_LOG"
echo "Inicio comparativa: $(date)" | tee -a "$MAIN_LOG"
echo "DATA_ROOT: $ALMENDROS_DATA_ROOT" | tee -a "$MAIN_LOG"
echo "RESULTS_ROOT: $ALMENDROS_RESULTS_ROOT" | tee -a "$MAIN_LOG"
echo "Espacio libre inicial: $(df -h / | awk 'NR==2 {print $4}')" | tee -a "$MAIN_LOG"
echo "Limpieza preventiva..." | tee -a "$MAIN_LOG"
cleanup_flower
echo "============================================" | tee -a "$MAIN_LOG"

run_strategy() {
    local NAME="$1"
    shift
    local CONFIG_STR="$*"
    local LOG_FILE="$LOG_DIR/${NAME}_${TIMESTAMP}.log"

    echo "" | tee -a "$MAIN_LOG"
    echo "→ [$NAME] Inicio: $(date +%H:%M:%S)" | tee -a "$MAIN_LOG"
    echo "  Config: $CONFIG_STR" | tee -a "$MAIN_LOG"
    echo "  Log: $LOG_FILE" | tee -a "$MAIN_LOG"

    local START_T=$(date +%s)
    flwr run . --stream --run-config "$CONFIG_STR" > "$LOG_FILE" 2>&1
    local EXIT_CODE=$?
    local END_T=$(date +%s)
    local DURATION=$((END_T - START_T))

    if [ $EXIT_CODE -eq 0 ] && [ $DURATION -gt 60 ]; then
        echo "  ✅ Completado en $((DURATION / 60))m $((DURATION % 60))s" | tee -a "$MAIN_LOG"
        local LAST_RUN=$(ls -td "$ALMENDROS_RESULTS_ROOT"/${NAME}_* 2>/dev/null | head -1)
        if [ -n "$LAST_RUN" ] && [ -n "$(find "$LAST_RUN" -mmin -$((DURATION/60+5)) -maxdepth 0 2>/dev/null)" ]; then
            echo "  📊 Generando summary de $(basename $LAST_RUN)..." | tee -a "$MAIN_LOG"
            python -m almendros_fl.summary "$LAST_RUN" 2>&1 | tee -a "$MAIN_LOG"
        else
            echo "  ⚠️  No se encontró carpeta de run reciente" | tee -a "$MAIN_LOG"
        fi
    else
        echo "  ❌ FALLO (exit=$EXIT_CODE, duración=${DURATION}s). Ver: $LOG_FILE" | tee -a "$MAIN_LOG"
        echo "  --- Últimas 50 líneas del log ---" | tee -a "$MAIN_LOG"
        tail -50 "$LOG_FILE" | tee -a "$MAIN_LOG"
    fi

    echo "  Espacio libre: $(df -h / | awk 'NR==2 {print $4}')" | tee -a "$MAIN_LOG"

    # Pausa para liberar recursos antes del siguiente run
    echo "  Pausa 15s antes del siguiente run..." | tee -a "$MAIN_LOG"
    sleep 15
}

run_strategy "FedAvg" 'strategy-name="FedAvg" seed=42 num-server-rounds=8'
run_strategy "FedProx" 'strategy-name="FedProx" proximal-mu=0.01 seed=42 num-server-rounds=8'
run_strategy "FedAvgM" 'strategy-name="FedAvgM" server-momentum=0.9 seed=42 num-server-rounds=8'

echo "" | tee -a "$MAIN_LOG"
echo "============================================" | tee -a "$MAIN_LOG"
echo "Fin: $(date)" | tee -a "$MAIN_LOG"
echo "============================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "📊 RESUMEN FINAL (última fila de cada summary):" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

for STRATEGY in FedAvg FedProx FedAvgM; do
    LAST_RUN=$(ls -td "$ALMENDROS_RESULTS_ROOT"/${STRATEGY}_* 2>/dev/null | head -1)
    if [ -n "$LAST_RUN" ] && [ -f "$LAST_RUN/summary.csv" ]; then
        echo "[$STRATEGY] $(basename $LAST_RUN)" | tee -a "$MAIN_LOG"
        head -1 "$LAST_RUN/summary.csv" | tee -a "$MAIN_LOG"
        tail -1 "$LAST_RUN/summary.csv" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
    else
        echo "[$STRATEGY] sin summary disponible" | tee -a "$MAIN_LOG"
    fi
done

echo "Log principal: $MAIN_LOG"
