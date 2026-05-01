"""Regenera server_metrics.csv combinando datos de client_metrics.csv y un dict
de global metrics extraído manualmente del log de consola."""

import csv
import sys
from pathlib import Path


# Datos extraídos manualmente de los logs de consola
GLOBAL_METRICS = {
    "FedAvg_20260426-112851": {
        # round: (global_loss, global_accuracy)
        0: (1.5124, 0.2800),
        1: (5.7811, 0.2200),
        2: (1.8804, 0.4225),
        3: (1.0865, 0.6275),
        4: (0.7400, 0.7475),
        5: (0.7373, 0.7300),
        6: (0.7029, 0.7450),
        7: (0.6553, 0.7650),
        8: (0.9643, 0.7300),
    },
    "Fedprox_20260426-142040": {
        0: (1.5124, 0.2800),
        1: (7.0801, 0.2200),
        2: (3.3866, 0.3025),
        3: (1.2810, 0.6200),
        4: (0.7398, 0.7575),
        5: (0.6292, 0.7675),
        6: (0.7233, 0.7550),
        7: (0.9302, 0.7300),
        8: (0.9088, 0.7450),
    },
}


def fix_run(run_dir: Path) -> None:
    name = run_dir.name
    if name not in GLOBAL_METRICS:
        print(f"[skip] {name} no está en GLOBAL_METRICS")
        return

    server_csv = run_dir / "server_metrics.csv"
    client_csv = run_dir / "client_metrics.csv"

    # Leer client_metrics y agrupar por ronda
    rounds: dict[int, list[dict]] = {}
    with open(client_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Saltar cabeceras duplicadas
            if row["round"] == "round":
                continue
            r = int(row["round"])
            rounds.setdefault(r, []).append(row)

    # Escribir nuevo server_metrics.csv
    out_path = run_dir / "server_metrics_fixed.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "global_loss", "global_accuracy",
            "train_loss_agg", "val_loss_agg", "val_accuracy_agg",
        ])

        for r in sorted(rounds.keys()):
            clients = rounds[r]
            total = sum(int(c["num_examples"]) for c in clients)
            train_loss_agg = sum(
                float(c["train_loss"]) * int(c["num_examples"]) for c in clients
            ) / total
            val_loss_agg = sum(
                float(c["val_loss"]) * int(c["num_examples"]) for c in clients
            ) / total
            val_acc_agg = sum(
                float(c["val_accuracy"]) * int(c["num_examples"]) for c in clients
            ) / total

            global_loss, global_acc = GLOBAL_METRICS[name].get(r, (None, None))
            writer.writerow([
                r,
                f"{global_loss:.6f}" if global_loss is not None else "",
                f"{global_acc:.6f}" if global_acc is not None else "",
                f"{train_loss_agg:.6f}",
                f"{val_loss_agg:.6f}",
                f"{val_acc_agg:.6f}",
            ])

    print(f"[ok] {name} → {out_path}")


if __name__ == "__main__":
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    for run_dir in sorted(results_root.iterdir()):
        if run_dir.is_dir():
            fix_run(run_dir)