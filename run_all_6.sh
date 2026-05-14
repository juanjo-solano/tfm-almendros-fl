#!/usr/bin/env bash
# run_all_6.sh — Tanda nocturna de 18 runs:
#   3 estrategias × 2 fases × 3 seeds = 18 runs MobileNetV2.
#
# El clúster Docker+TLS debe estar UP antes de lanzar este script.
#
# Uso:
#   ./run_all_6.sh                        # detecta ruta automáticamente
#   ALMENDROS_PROJECT=/otra/ruta ./run_all_6.sh  # override por entorno

set -e

# ── Detección de ruta del proyecto ───────────────────────────────────
# 1) Si está definida ALMENDROS_PROJECT, la usamos.
# 2) Si no, derivamos la ruta del propio script (run_all_6.sh vive en
#    la raíz del proyecto, junto a pyproject.toml).
if [ -n "${ALMENDROS_PROJECT:-}" ]; then
    PROJECT="$ALMENDROS_PROJECT"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT="$SCRIPT_DIR"
fi

if [ ! -f "$PROJECT/pyproject.toml" ]; then
    echo "[ERROR] No se encontró pyproject.toml en: $PROJECT"
    echo "        Define ALMENDROS_PROJECT=/ruta/al/proyecto y reintenta."
    exit 1
fi

echo "[Orquestador] Proyecto: $PROJECT"

CHECKPOINTS_DIR="$PROJECT/results/_phase1_checkpoints"
SERVER_CONTAINER="almendros_fl-superexec-serverapp-1"
LOG_DIR="$PROJECT/results/_run_logs"

mkdir -p "$CHECKPOINTS_DIR" "$LOG_DIR"
cd "$PROJECT"

# ── Seeds a ejecutar ─────────────────────────────────────────────────
SEEDS=(42 123 2024)

# ── Helpers ──────────────────────────────────────────────────────────
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
    # Copia el último final_model.pt del contenedor a un nombre identificable
    # por estrategia + seed. Esto evita que Fase 2 de seed=42 cargue por error
    # el checkpoint de seed=123.
    local strategy_name="$1"
    local seed="$2"
    local dest="$CHECKPOINTS_DIR/${strategy_name}_seed${seed}_phase1.pt"

    sleep 5
    LATEST_RUN=$(docker exec "$SERVER_CONTAINER" sh -c \
        "ls -td /app/results/*/ 2>/dev/null | head -1" | tr -d '\r')
    if [ -z "$LATEST_RUN" ]; then
        echo "[Orquestador] ERROR: no encontré el último run en el contenedor"
        return 1
    fi
    docker cp "$SERVER_CONTAINER:${LATEST_RUN}final_model.pt" "$dest"
    echo "[Orquestador] Checkpoint Fase 1 ($strategy_name, seed=$seed) -> $dest"
}

# ═══════════════════════════════════════════════════════════════════════
# Bucle principal: para cada seed, ejecutar las 6 configuraciones
# ═══════════════════════════════════════════════════════════════════════
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "#  SEED = $SEED"
    echo "############################################################"

    # ── 1. FedAvg Fase 1 ──────────────────────────────────────────
    run_one "1_FedAvg_phase1_seed${SEED}" \
        "num-server-rounds=12 strategy='fedavg' phase=1 seed=${SEED}"
    copy_phase1_checkpoint "FedAvg" "$SEED"

    # ── 2. FedAvg Fase 2 ──────────────────────────────────────────
    docker cp "$CHECKPOINTS_DIR/FedAvg_seed${SEED}_phase1.pt" \
        "$SERVER_CONTAINER:/app/init_FedAvg_seed${SEED}_phase1.pt"
    run_one "2_FedAvg_phase2_seed${SEED}" \
        "num-server-rounds=12 strategy='fedavg' phase=2 seed=${SEED} init-from='/app/init_FedAvg_seed${SEED}_phase1.pt'"

    # ── 3. FedProx Fase 1 ─────────────────────────────────────────
    run_one "3_FedProx_phase1_seed${SEED}" \
        "num-server-rounds=12 strategy='fedprox' phase=1 seed=${SEED} proximal-mu=0.001"
    copy_phase1_checkpoint "FedProx" "$SEED"

    # ── 4. FedProx Fase 2 ─────────────────────────────────────────
    docker cp "$CHECKPOINTS_DIR/FedProx_seed${SEED}_phase1.pt" \
        "$SERVER_CONTAINER:/app/init_FedProx_seed${SEED}_phase1.pt"
    run_one "4_FedProx_phase2_seed${SEED}" \
        "num-server-rounds=12 strategy='fedprox' phase=2 seed=${SEED} proximal-mu=0.001 init-from='/app/init_FedProx_seed${SEED}_phase1.pt'"

    # ── 5. FedBN Fase 1 ───────────────────────────────────────────
    # En Fase 1 FedBN se comporta como FedAvg (base congelada, sin BN entrenable).
    run_one "5_FedBN_phase1_seed${SEED}" \
        "num-server-rounds=12 strategy='fedbn' phase=1 seed=${SEED}"
    copy_phase1_checkpoint "FedBN" "$SEED"

    # ── 6. FedBN Fase 2 ───────────────────────────────────────────
    docker cp "$CHECKPOINTS_DIR/FedBN_seed${SEED}_phase1.pt" \
        "$SERVER_CONTAINER:/app/init_FedBN_seed${SEED}_phase1.pt"
    run_one "6_FedBN_phase2_seed${SEED}" \
        "num-server-rounds=12 strategy='fedbn' phase=2 seed=${SEED} init-from='/app/init_FedBN_seed${SEED}_phase1.pt'"
done

# ═══════════════════════════════════════════════════════════════════════
# Sacar TODOS los resultados al host
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TANDA COMPLETA: 18 runs ejecutados"
echo "  Sacando resultados al host..."
echo "════════════════════════════════════════════════════════════════"

OUT_DIR="$PROJECT/results/_docker_results_full_$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT_DIR"
docker cp "$SERVER_CONTAINER:/app/results/." "$OUT_DIR/"
echo "Resultados copiados a: $OUT_DIR"
echo "Hora final: $(date)"
echo ""
echo "Estructura:"
ls -la "$OUT_DIR" | head -25
