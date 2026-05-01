#!/usr/bin/env bash
# run_deployment.sh — versión 2: logs a archivo, sin depender de tmux para inspeccionar

set -euo pipefail

SESSION="almendros-fl"
SUPERLINK_ADDR="127.0.0.1:9092"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/deployment_logs"
NUM_SUPERNODES=4

export ALMENDROS_DATA_ROOT="${ALMENDROS_DATA_ROOT:-$PROJECT_DIR/data}"
export ALMENDROS_RESULTS_ROOT="${ALMENDROS_RESULTS_ROOT:-$PROJECT_DIR/results}"

case "${1:-}" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[!] La sesión '$SESSION' ya existe. Usa 'stop' antes de arrancar otra vez."
      exit 1
    fi

    mkdir -p "$LOG_DIR"
    echo "[*] Logs en: $LOG_DIR"
    echo "[*] Creando sesión tmux '$SESSION'..."
    tmux new-session -d -s "$SESSION" -n superlink

    # Ventana 0: SuperLink (con tee a archivo)
    tmux send-keys -t "$SESSION:superlink" \
      "cd '$PROJECT_DIR' && flower-superlink --insecure 2>&1 | tee '$LOG_DIR/superlink.log'" C-m

    # Ventanas 1..N: SuperNodes
for i in $(seq 0 $((NUM_SUPERNODES - 1))); do
      tmux new-window -t "$SESSION" -n "supernode-$i"
      CLIENTAPPIO_PORT=$((9094 + i))
      tmux send-keys -t "$SESSION:supernode-$i" \
        "cd '$PROJECT_DIR' && \
         export ALMENDROS_DATA_ROOT='$ALMENDROS_DATA_ROOT' && \
         sleep 3 && \
         flower-supernode \
           --insecure \
           --superlink $SUPERLINK_ADDR \
           --clientappio-api-address 127.0.0.1:$CLIENTAPPIO_PORT \
           --node-config 'partition-id=$i' 2>&1 | tee '$LOG_DIR/supernode-$i.log'" C-m
    done

    echo "[OK] Clúster iniciándose. Espera ~5 segundos y revisa con:"
    echo "    ./run_deployment.sh logs       # ver últimas líneas de todos"
    echo "    ./run_deployment.sh log superlink"
    echo "    ./run_deployment.sh log 0      # supernode-0"
    ;;

  run)
    cd "$PROJECT_DIR"
    flwr run . local-deployment --stream
    ;;

  logs)
    if [ ! -d "$LOG_DIR" ]; then
      echo "[!] No hay logs en $LOG_DIR. ¿Has hecho 'start'?"
      exit 1
    fi
    for f in "$LOG_DIR"/*.log; do
      echo ""
      echo "═══════════════════════════════════════════════════════════"
      echo "  $(basename "$f")"
      echo "═══════════════════════════════════════════════════════════"
      tail -n 30 "$f"
    done
    ;;

  log)
    name="${2:-}"
    if [ -z "$name" ]; then
      echo "Uso: $0 log {superlink|0|1|2|3}"
      exit 1
    fi
    case "$name" in
      superlink) f="$LOG_DIR/superlink.log" ;;
      [0-3])     f="$LOG_DIR/supernode-$name.log" ;;
      *)         f="$LOG_DIR/$name.log" ;;
    esac
    if [ ! -f "$f" ]; then
      echo "[!] No existe: $f"
      ls "$LOG_DIR/" 2>/dev/null || echo "(sin logs aún)"
      exit 1
    fi
    cat "$f"
    ;;

  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux kill-session -t "$SESSION"
      echo "[OK] Sesión tmux terminada."
    fi
    pkill -f flower-superlink 2>/dev/null || true
    pkill -f flower-supernode 2>/dev/null || true
    pkill -f flower-superexec 2>/dev/null || true
    echo "[OK] Procesos Flower limpiados."
    ;;

  status)
    echo "[*] Procesos Flower vivos:"
    ps -ef | grep -E "flower-superlink|flower-supernode" | grep -v grep || echo "  (ninguno)"
    echo ""
    echo "[*] Logs disponibles:"
    ls -la "$LOG_DIR/" 2>/dev/null || echo "  (no hay logs)"
    ;;

  *)
    echo "Uso: $0 {start|run|logs|log NAME|stop|status}"
    echo ""
    echo "  start         Levanta SuperLink + 4 SuperNodes"
    echo "  run           Lanza un run federado"
    echo "  logs          Muestra las últimas 30 líneas de TODOS los logs"
    echo "  log NAME      Muestra log completo (NAME = superlink|0|1|2|3)"
    echo "  stop          Mata todo"
    echo "  status        Muestra procesos vivos"
    exit 1
    ;;
esac