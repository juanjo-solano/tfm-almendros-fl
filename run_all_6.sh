#!/usr/bin/env bash
# run_all_6.sh — Tanda nocturna de 6 runs MobileNetV2 (3 estrategias × 2 fases).
# El clúster Docker+TLS debe estar UP antes de lanzar este script.

set -e

PROJECT="$HOME/TFM/Flower_ejemplo/almendros/almendros_fl"
CHECKPOINTS_DIR="$PROJECT/results/_phase1_checkpoints"
SERVER_CONTAINER="almendros_fl-superexec-serverapp-1"
LOG_DIR="$PROJECT/results/_run_logs"

mkdir -p "$CHECKPOINTS_DIR" "$LOG_DIR"
cd "$PROJECT"

run_one () {
    local run_name="$1"
    local extra_config="$2"
    local logfile="$LOG_DIR/${run_name}_$(date +%Y%m%d-%H%M%S).log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Lanzando: $run_name"
    echo "  Config:   $extra_config"
    echo "  Log:      $logfile"
    echo "  Hora:     $(date)"
    echo "════════════════════════════════════════════════════════════════"
    flwr run . local-deployment-tls --stream \
        --run-config "$extra_config" 2>&1 | tee "$logfile"
}

copy_phase1_checkpoint () {
    local strategy_name="$1"
    local dest="$CHECKPOINTS_DIR/${strategy_name}_phase1.pt"
    sleep 5
    LATEST_RUN=$(docker exec "$SERVER_CONTAINER" sh -c \
        "ls -td /app/results/*/ 2>/dev/null | head -1" | tr -d '\r')
    if [ -z "$LATEST_RUN" ]; then
        echo "[Orquestador] ERROR: no encontre el ultimo run en el contenedor"
        return 1
    fi
    docker cp "$SERVER_CONTAINER:${LATEST_RUN}final_model.pt" "$dest"
    echo "[Orquestador] Checkpoint Fase 1 ($strategy_name) -> $dest"
}

# ═══════════════════════════════════════════════════════════════════════
# 1. FedAvg Fase 1
# ═══════════════════════════════════════════════════════════════════════
run_one "1_FedAvg_phase1" "num-server-rounds=12 strategy='fedavg' phase=1"
copy_phase1_checkpoint "FedAvg"

# ═══════════════════════════════════════════════════════════════════════
# 2. FedAvg Fase 2 (init-from FedAvg_phase1.pt)
# ═══════════════════════════════════════════════════════════════════════
docker cp "$CHECKPOINTS_DIR/FedAvg_phase1.pt" \
    "$SERVER_CONTAINER:/app/init_FedAvg_phase1.pt"
run_one "2_FedAvg_phase2" "num-server-rounds=12 strategy='fedavg' phase=2 init-from='/app/init_FedAvg_phase1.pt'"

# ═══════════════════════════════════════════════════════════════════════
# 3. FedProx Fase 1
# ═══════════════════════════════════════════════════════════════════════
run_one "3_FedProx_phase1" "num-server-rounds=12 strategy='fedprox' phase=1 proximal-mu=0.001"
copy_phase1_checkpoint "FedProx"

# ═══════════════════════════════════════════════════════════════════════
# 4. FedProx Fase 2
# ═══════════════════════════════════════════════════════════════════════
docker cp "$CHECKPOINTS_DIR/FedProx_phase1.pt" \
    "$SERVER_CONTAINER:/app/init_FedProx_phase1.pt"
run_one "4_FedProx_phase2" "num-server-rounds=12 strategy='fedprox' phase=2 proximal-mu=0.001 init-from='/app/init_FedProx_phase1.pt'"

# ═══════════════════════════════════════════════════════════════════════
# 5. FedBN Fase 1
# ═══════════════════════════════════════════════════════════════════════
# Nota: FedBN en Fase 1 no tiene BN entrenable (base congelada), se comportara
# como FedAvg. Lo lanzamos igual por completitud.
run_one "5_FedBN_phase1" "num-server-rounds=12 strategy='fedbn' phase=1"
copy_phase1_checkpoint "FedBN"

# ═══════════════════════════════════════════════════════════════════════
# 6. FedBN Fase 2 (aqui FedBN si tiene efecto: BN del fine-tuning son locales)
# ═══════════════════════════════════════════════════════════════════════
docker cp "$CHECKPOINTS_DIR/FedBN_phase1.pt" \
    "$SERVER_CONTAINER:/app/init_FedBN_phase1.pt"
run_one "6_FedBN_phase2" "num-server-rounds=12 strategy='fedbn' phase=2 init-from='/app/init_FedBN_phase1.pt'"

# ═══════════════════════════════════════════════════════════════════════
# Sacar TODOS los resultados al host
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TANDA COMPLETA"
echo "  Sacando resultados al host..."
echo "════════════════════════════════════════════════════════════════"

OUT_DIR="$PROJECT/results/_docker_results_full_$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT_DIR"
docker cp "$SERVER_CONTAINER:/app/results/." "$OUT_DIR/"
echo "Resultados copiados a: $OUT_DIR"
echo "Hora final: $(date)"
echo ""
echo "Estructura:"
ls -la "$OUT_DIR" | head -20
