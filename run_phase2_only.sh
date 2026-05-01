#!/usr/bin/env bash
# run_phase2_only.sh — Relanza solo las 3 Fases 2 con init-from corregido.

set -e

PROJECT="$HOME/TFM/Flower_ejemplo/almendros/almendros_fl"
CHECKPOINTS_DIR="$PROJECT/results/_phase1_checkpoints"
SERVER_CONTAINER="almendros_fl-superexec-serverapp-1"
LOG_DIR="$PROJECT/results/_run_logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT"

run_one () {
    local run_name="$1"
    local extra_config="$2"
    local logfile="$LOG_DIR/${run_name}_$(date +%Y%m%d-%H%M%S).log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Lanzando: $run_name"
    echo "  Config:   $extra_config"
    echo "  Hora:     $(date)"
    echo "════════════════════════════════════════════════════════════════"
    flwr run . local-deployment-tls --stream \
        --run-config "$extra_config" 2>&1 | tee "$logfile"
}

# 2. FedAvg Fase 2
docker cp "$CHECKPOINTS_DIR/FedAvg_phase1.pt" \
    "$SERVER_CONTAINER:/app/init_FedAvg_phase1.pt"
run_one "2_FedAvg_phase2_v2" "num-server-rounds=12 strategy='fedavg' phase=2 init-from='/app/init_FedAvg_phase1.pt'"

# 4. FedProx Fase 2
docker cp "$CHECKPOINTS_DIR/FedProx_phase1.pt" \
    "$SERVER_CONTAINER:/app/init_FedProx_phase1.pt"
run_one "4_FedProx_phase2_v2" "num-server-rounds=12 strategy='fedprox' phase=2 proximal-mu=0.001 init-from='/app/init_FedProx_phase1.pt'"

# 6. FedBN Fase 2
docker cp "$CHECKPOINTS_DIR/FedBN_phase1.pt" \
    "$SERVER_CONTAINER:/app/init_FedBN_phase1.pt"
run_one "6_FedBN_phase2_v2" "num-server-rounds=12 strategy='fedbn' phase=2 init-from='/app/init_FedBN_phase1.pt'"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  3 FASES 2 COMPLETADAS"
echo "════════════════════════════════════════════════════════════════"

OUT_DIR="$PROJECT/results/_docker_results_phase2_$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT_DIR"
docker cp "$SERVER_CONTAINER:/app/results/." "$OUT_DIR/"
echo "Resultados copiados a: $OUT_DIR"
echo "Hora final: $(date)"
