"""Genera summary.csv unificado a partir de server_metrics.csv y client_metrics.csv.

Uso:
    python -m almendros_fl.summary results/FedAvg_YYYYMMDD-HHMMSS
    python -m almendros_fl.summary results/FedAvg_YYYYMMDD-HHMMSS --print
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    """Promedio ponderado, ignorando NaN."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    v = values[mask]
    w = weights[mask]
    return float((v * w).sum() / w.sum())


def make_summary(run_dir: Path) -> pd.DataFrame:
    """Combina los CSVs por cliente y server en un summary unificado."""
    server_csv = run_dir / "server_metrics.csv"
    client_csv = run_dir / "client_metrics.csv"

    if not server_csv.exists() or not client_csv.exists():
        raise FileNotFoundError(
            f"Faltan CSVs en {run_dir}. Esperaba: {server_csv.name} y {client_csv.name}"
        )

    server = pd.read_csv(server_csv)
    client = pd.read_csv(client_csv)

    # Agregados weighted-by-num_examples por ronda
    agg = (
        client.groupby("round", group_keys=True)
        .apply(
            lambda g: pd.Series({
                "train_loss_agg": _weighted_avg(g["train_loss"], g["num_examples"]),
                "val_loss_agg": _weighted_avg(g["val_loss"], g["num_examples"]),
                "val_accuracy_agg": _weighted_avg(g["val_accuracy"], g["num_examples"]),
                "num_clients": g["partition_id"].nunique(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    # Eliminar columnas vacías de server antes del merge para no duplicar
    cols_to_drop = [c for c in
                    ["train_loss_agg", "val_loss_agg", "val_accuracy_agg"]
                    if c in server.columns]
    server_clean = server.drop(columns=cols_to_drop)

    summary = server_clean.merge(agg, on="round", how="left")
    return summary


def write_summary(run_dir: Path, summary: pd.DataFrame) -> Path:
    """Escribe summary.csv en la carpeta del run."""
    out = run_dir / "summary.csv"
    summary.to_csv(out, index=False, float_format="%.6f")
    return out


def main():
    parser = argparse.ArgumentParser(description="Genera summary.csv de un run FL.")
    parser.add_argument("run_dir", type=Path, help="Carpeta del run")
    parser.add_argument("--print", action="store_true",
                        help="Imprime el summary por pantalla")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"[summary] No existe la carpeta: {run_dir}", file=sys.stderr)
        sys.exit(1)

    summary = make_summary(run_dir)
    out = write_summary(run_dir, summary)
    print(f"[summary] Escrito en: {out}")

    if args.print:
        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()