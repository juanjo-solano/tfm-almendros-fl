"""aggregate_seeds.py — Agrega resultados de runs multi-seed.

Lee todas las carpetas results/<Strategy>_seed<N>_<timestamp>/ y produce un
CSV resumen con la media ± desviación estándar de la accuracy final
(global_accuracy de la última ronda) para cada (estrategia, fase).

Uso:
    python aggregate_seeds.py
    python aggregate_seeds.py --results-dir results/_docker_results_full_20260515-0830
"""

import argparse
import re
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


def parse_run_dir(run_dir: Path) -> dict | None:
    """Extrae (strategy, seed, phase) leyendo el config.txt del run."""
    config_path = run_dir / "config.txt"
    if not config_path.exists():
        return None

    info = {}
    with open(config_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                info[k] = v

    strategy = info.get("strategy", "Unknown")
    # Algunas claves vienen con guion, otras con underscore
    seed = info.get("seed")
    phase = info.get("phase")

    if seed is None or phase is None:
        return None

    return {
        "run_dir": run_dir,
        "strategy": strategy,
        "seed": int(seed),
        "phase": int(phase),
    }


def get_final_metrics(run_dir: Path) -> dict | None:
    """Lee la última fila de server_metrics.csv (= métricas finales)."""
    csv_path = run_dir / "server_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    last = df.iloc[-1]
    return {
        "global_accuracy": last.get("global_accuracy"),
        "global_loss": last.get("global_loss"),
        "val_accuracy_agg": last.get("val_accuracy_agg"),
        "num_rounds": int(last.get("round", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="results",
        help="Directorio con las carpetas de runs (default: results/)",
    )
    parser.add_argument(
        "--output", default="results/summary_multiseed.csv",
        help="Ruta del CSV resumen (default: results/summary_multiseed.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"No existe: {results_dir}")

    # Descubrir todos los runs
    all_runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("_"):
            continue  # saltar _phase1_checkpoints, _run_logs, etc.
        info = parse_run_dir(run_dir)
        if info is None:
            continue
        metrics = get_final_metrics(run_dir)
        if metrics is None:
            continue
        all_runs.append({**info, **metrics})

    if not all_runs:
        print(f"[WARN] No se encontraron runs válidos en {results_dir}")
        return

    df = pd.DataFrame(all_runs)
    print(f"[OK] Encontrados {len(df)} runs.")
    print()

    # Agrupar por (strategy, phase) y calcular media/std
    grouped = df.groupby(["strategy", "phase"]).agg(
        n_seeds=("seed", "count"),
        seeds=("seed", lambda s: sorted(s.tolist())),
        acc_mean=("global_accuracy", "mean"),
        acc_std=("global_accuracy", "std"),
        loss_mean=("global_loss", "mean"),
        loss_std=("global_loss", "std"),
    ).reset_index()

    # Formato bonito para la consola
    print("=" * 80)
    print(f"{'Strategy':<12} {'Phase':<6} {'N':<3} {'Seeds':<20} "
          f"{'Acc (mean ± std)':<22} {'Loss (mean ± std)':<22}")
    print("=" * 80)
    for _, row in grouped.iterrows():
        acc_str = f"{row['acc_mean']:.4f} ± {row['acc_std']:.4f}"
        loss_str = f"{row['loss_mean']:.4f} ± {row['loss_std']:.4f}"
        seeds_str = str(row["seeds"])
        print(f"{row['strategy']:<12} {row['phase']:<6} {row['n_seeds']:<3} "
              f"{seeds_str:<20} {acc_str:<22} {loss_str:<22}")
    print("=" * 80)

    # Guardar CSV
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    print(f"\n[OK] Resumen guardado en: {output_path}")

    # Guardar también el detalle por run
    detail_path = output_path.parent / "summary_multiseed_detail.csv"
    df.sort_values(["strategy", "phase", "seed"]).to_csv(detail_path, index=False)
    print(f"[OK] Detalle por run guardado en: {detail_path}")


if __name__ == "__main__":
    main()