#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_figures.py — Genera todas las gráficas del TFM desde los CSVs de
los runs guardados en results/.

Ejecución:
    cd ~/TFM/Flower_ejemplo/almendros/almendros_fl
    python3 generate_figures.py

Genera PNGs en:
    figs/
        01_curvas_aprendizaje_resnet18.png
        02_curvas_aprendizaje_mobilenetv2_F1.png
        03_curvas_aprendizaje_mobilenetv2_F2.png
        04_comparativa_resnet18_vs_mobilenetv2.png
        05_brecha_entre_clientes.png
        06_fase1_vs_fase2.png
        07_train_loss_vs_val_acc_overconfidence.png

Requisitos:
    pip install pandas matplotlib

El script es robusto: si falta algún CSV, se salta esa figura con aviso.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
OUT = PROJECT / "figs"
OUT.mkdir(exist_ok=True)

# Colores consistentes para las estrategias
COLORS = {
    "FedAvg":  "#1f77b4",   # azul
    "FedProx": "#2ca02c",   # verde
    "FedBN":   "#d62728",   # rojo
    "FedAvgM": "#ff7f0e",   # naranja
}
LINESTYLES = {
    "F1": "--",
    "F2": "-",
    "":   "-",
}

# Estilo global
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────
# Mapeo de carpetas a (label, arquitectura, estrategia, fase, runtime)
# ─────────────────────────────────────────────────────────────────────
RUNS = [
    # ─── ResNet18 ───────────────────────────────────────────────
    {
        "path": "FedAvg_20260426-112851",
        "label": "FedAvg (ResNet18, sim)",
        "arch": "ResNet18", "strategy": "FedAvg",
        "phase": "", "runtime": "Simulación",
    },
    {
        "path": "Fedprox_20260426-142040",
        "label": "FedProx μ=0.01 (ResNet18, sim)",
        "arch": "ResNet18", "strategy": "FedProx",
        "phase": "", "runtime": "Simulación",
    },
    {
        "path": "Fedbn_20260426-162815",
        "label": "FedBN (ResNet18, sim)",
        "arch": "ResNet18", "strategy": "FedBN",
        "phase": "", "runtime": "Simulación",
    },
    {
        "path": "Fedprox_20260427-125231-094",
        "label": "FedProx (ResNet18, deployment local)",
        "arch": "ResNet18", "strategy": "FedProx",
        "phase": "", "runtime": "Deployment local",
    },
    {
        "path": "Fedprox_DOCKER_20260428-085954-079",
        "label": "FedProx (ResNet18, Docker)",
        "arch": "ResNet18", "strategy": "FedProx",
        "phase": "", "runtime": "Docker",
    },
    # ─── MobileNetV2 Fase 1 (Docker+TLS) ────────────────────────
    {
        "path": "_docker_results_full_20260429-0443/Fedavg_20260428-214855-007",
        "label": "FedAvg (MobileNetV2 F1)",
        "arch": "MobileNetV2", "strategy": "FedAvg",
        "phase": "F1", "runtime": "Docker+TLS",
    },
    {
        "path": "_docker_results_full_20260429-0443/Fedprox_20260428-232700-387",
        "label": "FedProx (MobileNetV2 F1)",
        "arch": "MobileNetV2", "strategy": "FedProx",
        "phase": "F1", "runtime": "Docker+TLS",
    },
    {
        "path": "_docker_results_full_20260429-0443/Fedbn_20260429-010542-065",
        "label": "FedBN (MobileNetV2 F1)",
        "arch": "MobileNetV2", "strategy": "FedBN",
        "phase": "F1", "runtime": "Docker+TLS",
    },
    # ─── MobileNetV2 Fase 2 (Docker+TLS) ────────────────────────
    {
        "path": "_docker_results_phase2_20260429-1406/Fedavg_20260429-083224-247",
        "label": "FedAvg (MobileNetV2 F2)",
        "arch": "MobileNetV2", "strategy": "FedAvg",
        "phase": "F2", "runtime": "Docker+TLS",
    },
    {
        "path": "_docker_results_phase2_20260429-1406/Fedprox_20260429-094241-767",
        "label": "FedProx (MobileNetV2 F2) ★ GANADOR",
        "arch": "MobileNetV2", "strategy": "FedProx",
        "phase": "F2", "runtime": "Docker+TLS",
    },
    {
        "path": "_docker_results_phase2_20260429-1406/Fedbn_20260429-105546-338",
        "label": "FedBN (MobileNetV2 F2)",
        "arch": "MobileNetV2", "strategy": "FedBN",
        "phase": "F2", "runtime": "Docker+TLS",
    },
]

# Si añadiste el run de Fase 6.4 (Docker+TLS con ResNet18), inclúyelo aquí:
# {
#     "path": "Fedprox_DOCKER_TLS_20260428-103910-159",
#     "label": "FedProx (ResNet18, Docker+TLS)",
#     "arch": "ResNet18", "strategy": "FedProx",
#     "phase": "", "runtime": "Docker+TLS",
# },


def load_run(run_meta):
    """Carga server_metrics.csv y client_metrics.csv de un run.
    Devuelve (server_df, client_df) o (None, None) si no existe."""
    run_dir = RESULTS / run_meta["path"]
    server_csv = run_dir / "server_metrics.csv"
    client_csv = run_dir / "client_metrics.csv"

    if not server_csv.exists():
        print(f"  [skip] No existe {server_csv}")
        return None, None

    server_df = pd.read_csv(server_csv)
    client_df = pd.read_csv(client_csv) if client_csv.exists() else None
    return server_df, client_df


def get_color_and_style(run_meta):
    color = COLORS.get(run_meta["strategy"], "gray")
    style = LINESTYLES.get(run_meta["phase"], "-")
    return color, style


# ─────────────────────────────────────────────────────────────────────
# Figura 1: Curvas de aprendizaje ResNet18 (simulación)
# ─────────────────────────────────────────────────────────────────────
def fig_01_curvas_resnet18():
    print("→ Figura 1: curvas de aprendizaje ResNet18 (simulación)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for run in RUNS:
        if run["arch"] != "ResNet18" or run["runtime"] != "Simulación":
            continue
        server_df, _ = load_run(run)
        if server_df is None:
            continue

        # Eje X: la columna 'round' o el índice
        if "round" in server_df.columns:
            x = server_df["round"]
        else:
            x = server_df.index

        color = COLORS.get(run["strategy"], "gray")
        label = f'{run["strategy"]}'

        # global_accuracy
        if "global_accuracy" in server_df.columns:
            axes[0].plot(x, server_df["global_accuracy"], "-o",
                         color=color, label=label, linewidth=2)
        # global_loss
        if "global_loss" in server_df.columns:
            axes[1].plot(x, server_df["global_loss"], "-o",
                         color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("Accuracy global")
    axes[0].set_title("Accuracy global — ResNet18 (simulación)")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("Loss global")
    axes[1].set_title("Loss global — ResNet18 (simulación)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUT / "01_curvas_aprendizaje_resnet18.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 2: Curvas MobileNetV2 Fase 1
# ─────────────────────────────────────────────────────────────────────
def fig_02_curvas_mobilenetv2_F1():
    print("→ Figura 2: curvas de aprendizaje MobileNetV2 Fase 1")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for run in RUNS:
        if run["arch"] != "MobileNetV2" or run["phase"] != "F1":
            continue
        server_df, _ = load_run(run)
        if server_df is None:
            continue

        x = server_df["round"] if "round" in server_df.columns else server_df.index
        color = COLORS.get(run["strategy"], "gray")
        label = run["strategy"]

        if "global_accuracy" in server_df.columns:
            axes[0].plot(x, server_df["global_accuracy"], "-o",
                         color=color, label=label, linewidth=2)
        if "global_loss" in server_df.columns:
            axes[1].plot(x, server_df["global_loss"], "-o",
                         color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("Accuracy global")
    axes[0].set_title("Accuracy global — MobileNetV2 Fase 1 (base congelada)")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("Loss global")
    axes[1].set_title("Loss global — MobileNetV2 Fase 1")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUT / "02_curvas_aprendizaje_mobilenetv2_F1.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 3: Curvas MobileNetV2 Fase 2
# ─────────────────────────────────────────────────────────────────────
def fig_03_curvas_mobilenetv2_F2():
    print("→ Figura 3: curvas de aprendizaje MobileNetV2 Fase 2")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for run in RUNS:
        if run["arch"] != "MobileNetV2" or run["phase"] != "F2":
            continue
        server_df, _ = load_run(run)
        if server_df is None:
            continue

        x = server_df["round"] if "round" in server_df.columns else server_df.index
        color = COLORS.get(run["strategy"], "gray")
        label = run["strategy"]

        if "global_accuracy" in server_df.columns:
            axes[0].plot(x, server_df["global_accuracy"], "-o",
                         color=color, label=label, linewidth=2)
        if "global_loss" in server_df.columns:
            axes[1].plot(x, server_df["global_loss"], "-o",
                         color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Ronda")
    axes[0].set_ylabel("Accuracy global")
    axes[0].set_title("Accuracy global — MobileNetV2 Fase 2 (fine-tuning)")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Ronda")
    axes[1].set_ylabel("Loss global")
    axes[1].set_title("Loss global — MobileNetV2 Fase 2")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUT / "03_curvas_aprendizaje_mobilenetv2_F2.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 4: Comparativa ResNet18 vs MobileNetV2 (FedProx)
# ─────────────────────────────────────────────────────────────────────
def fig_04_comparativa_arch():
    print("→ Figura 4: comparativa ResNet18 vs MobileNetV2 (FedProx)")
    fig, ax = plt.subplots(figsize=(10, 6))

    runs_to_compare = [
        ("Fedprox_20260426-142040", "FedProx ResNet18 (sim)", "#1f77b4", "--"),
        ("Fedprox_20260427-125231-094", "FedProx ResNet18 (deployment)", "#1f77b4", ":"),
        ("Fedprox_DOCKER_20260428-085954-079", "FedProx ResNet18 (Docker)", "#1f77b4", "-."),
        ("_docker_results_full_20260429-0443/Fedprox_20260428-232700-387",
         "FedProx MobileNetV2 F1", "#2ca02c", "--"),
        ("_docker_results_phase2_20260429-1406/Fedprox_20260429-094241-767",
         "FedProx MobileNetV2 F2 ★", "#2ca02c", "-"),
    ]

    for path, label, color, style in runs_to_compare:
        csv = RESULTS / path / "server_metrics.csv"
        if not csv.exists():
            print(f"  [skip] {csv}")
            continue
        df = pd.read_csv(csv)
        x = df["round"] if "round" in df.columns else df.index
        ax.plot(x, df["global_accuracy"], style + "o",
                color=color, label=label, linewidth=2,
                markersize=6)

    ax.set_xlabel("Ronda")
    ax.set_ylabel("Accuracy global")
    ax.set_title("Comparativa: FedProx en ResNet18 vs MobileNetV2 (todos los runtimes)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.axhline(0.875, color="green", linestyle=":", alpha=0.5,
               label="_target")
    ax.text(0.02, 0.86, "Mejor MobileNetV2 F2: 0.875",
            transform=ax.get_yaxis_transform(),
            color="green", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT / "04_comparativa_resnet18_vs_mobilenetv2.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 5: Brecha entre clientes (mejor run, MobileNetV2 FedProx F2)
# ─────────────────────────────────────────────────────────────────────
def fig_05_brecha_clientes():
    print("→ Figura 5: brecha entre clientes (run ganador)")
    csv = RESULTS / "_docker_results_phase2_20260429-1406" / \
                    "Fedprox_20260429-094241-767" / "client_metrics.csv"
    if not csv.exists():
        print(f"  [skip] No existe {csv}")
        return

    df = pd.read_csv(csv)
    print(f"  Columnas: {list(df.columns)}")

    # Esperamos columnas tipo: round, partition_id, val_accuracy, val_loss, ...
    # Si el formato es distinto, adaptamos.
    fig, ax = plt.subplots(figsize=(10, 6))

    contexts = ["manana", "tarde", "nublado", "otros_moviles"]
    colors_ctx = {"manana": "#FFA500", "tarde": "#8B4513",
                  "nublado": "#808080", "otros_moviles": "#4B0082"}

    # Detectar columna de partition_id
    pid_col = None
    for cand in ["partition_id", "_partition_id", "client_id"]:
        if cand in df.columns:
            pid_col = cand
            break

    if pid_col is None:
        # Si no hay columna partition, intentar con context
        if "context" in df.columns:
            for ctx in contexts:
                sub = df[df["context"] == ctx]
                if len(sub) > 0 and "val_accuracy" in sub.columns:
                    x = sub["round"] if "round" in sub.columns else sub.index
                    ax.plot(x, sub["val_accuracy"], "-o",
                            color=colors_ctx[ctx], label=ctx, linewidth=2)
        else:
            print("  [skip] No encuentro columnas de cliente en client_metrics.csv")
            plt.close(fig)
            return
    else:
        for i, ctx in enumerate(contexts):
            sub = df[df[pid_col] == i]
            if len(sub) == 0:
                continue
            x = sub["round"] if "round" in sub.columns else range(len(sub))
            if "val_accuracy" in sub.columns:
                ax.plot(x, sub["val_accuracy"], "-o",
                        color=colors_ctx[ctx], label=f"Cliente {i} ({ctx})",
                        linewidth=2)

    ax.set_xlabel("Ronda")
    ax.set_ylabel("Accuracy de validación")
    ax.set_title("Brecha entre clientes — MobileNetV2 FedProx Fase 2 (mejor run)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(OUT / "05_brecha_entre_clientes.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 6: Comparativa Fase 1 vs Fase 2 (las 3 estrategias)
# ─────────────────────────────────────────────────────────────────────
def fig_06_fase1_vs_fase2():
    print("→ Figura 6: comparativa Fase 1 vs Fase 2")
    fig, ax = plt.subplots(figsize=(10, 6))

    pairs = [
        ("FedAvg",
         "_docker_results_full_20260429-0443/Fedavg_20260428-214855-007",
         "_docker_results_phase2_20260429-1406/Fedavg_20260429-083224-247"),
        ("FedProx",
         "_docker_results_full_20260429-0443/Fedprox_20260428-232700-387",
         "_docker_results_phase2_20260429-1406/Fedprox_20260429-094241-767"),
        ("FedBN",
         "_docker_results_full_20260429-0443/Fedbn_20260429-010542-065",
         "_docker_results_phase2_20260429-1406/Fedbn_20260429-105546-338"),
    ]

    for strategy, p1, p2 in pairs:
        c = COLORS[strategy]
        csv1 = RESULTS / p1 / "server_metrics.csv"
        csv2 = RESULTS / p2 / "server_metrics.csv"
        if csv1.exists():
            df1 = pd.read_csv(csv1)
            x1 = df1["round"] if "round" in df1.columns else df1.index
            ax.plot(x1, df1["global_accuracy"], "--o",
                    color=c, label=f"{strategy} F1",
                    linewidth=2, alpha=0.7)
        if csv2.exists():
            df2 = pd.read_csv(csv2)
            x2 = df2["round"] if "round" in df2.columns else df2.index
            # Desplazamos las rondas de F2 para que se vea la continuidad
            x2_shifted = x2 + 12 if "round" in df2.columns else \
                         range(len(df1) if csv1.exists() else 0,
                               len(df1) + len(df2) if csv1.exists() else len(df2))
            ax.plot(x2_shifted, df2["global_accuracy"], "-o",
                    color=c, label=f"{strategy} F2 (fine-tuning)",
                    linewidth=2.5)

    # Línea vertical separando fases
    ax.axvline(12, color="black", linestyle=":", alpha=0.5)
    ax.text(12.2, 0.05, "← Fase 1 | Fase 2 →",
            color="black", fontsize=10)

    ax.set_xlabel("Ronda (acumulada)")
    ax.set_ylabel("Accuracy global")
    ax.set_title("MobileNetV2: Fase 1 (base congelada) → Fase 2 (fine-tuning)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(OUT / "06_fase1_vs_fase2.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 7: Overconfidence (train_loss vs val_acc en Fase 2)
# ─────────────────────────────────────────────────────────────────────
def fig_07_overconfidence():
    print("→ Figura 7: fenómeno de overconfidence en Fase 2")
    csv = RESULTS / "_docker_results_phase2_20260429-1406" / \
                    "Fedprox_20260429-094241-767" / "server_metrics.csv"
    if not csv.exists():
        print(f"  [skip] No existe {csv}")
        return

    df = pd.read_csv(csv)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = df["round"] if "round" in df.columns else df.index
    color1 = "tab:blue"
    color2 = "tab:red"

    ax1.set_xlabel("Ronda")
    ax1.set_ylabel("Accuracy global", color=color1)
    if "global_accuracy" in df.columns:
        ax1.plot(x, df["global_accuracy"], "-o",
                 color=color1, linewidth=2, label="Accuracy global")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.5, 1.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss global", color=color2)
    if "global_loss" in df.columns:
        ax2.plot(x, df["global_loss"], "-s",
                 color=color2, linewidth=2, label="Loss global")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.spines["top"].set_visible(False)

    plt.title("Overconfidence en Fase 2 — Accuracy SUBE pero Loss también SUBE\n"
              "(MobileNetV2 FedProx F2)")
    fig.tight_layout()
    fig.savefig(OUT / "07_train_loss_vs_val_acc_overconfidence.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figura 8: Tabla resumen como gráfica de barras
# ─────────────────────────────────────────────────────────────────────
def fig_08_tabla_resumen():
    print("→ Figura 8: tabla resumen (gráfica de barras)")
    data = []
    for run in RUNS:
        server_df, _ = load_run(run)
        if server_df is None or "global_accuracy" not in server_df.columns:
            continue
        peak = server_df["global_accuracy"].max()
        last = server_df["global_accuracy"].iloc[-1]
        data.append({
            "label": run["label"][:40],  # truncar
            "peak": peak,
            "last": last,
            "color": COLORS.get(run["strategy"], "gray"),
        })

    if not data:
        print("  [skip] No hay datos para la tabla resumen")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.5)))
    labels = [d["label"] for d in data]
    peaks = [d["peak"] for d in data]
    lasts = [d["last"] for d in data]
    colors = [d["color"] for d in data]

    y_pos = range(len(labels))
    bar_height = 0.35

    ax.barh([y - bar_height/2 for y in y_pos], peaks, bar_height,
            label="Pico", color=colors, alpha=0.9, edgecolor="black")
    ax.barh([y + bar_height/2 for y in y_pos], lasts, bar_height,
            label="Final R12/R8", color=colors, alpha=0.5, edgecolor="black")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy global")
    ax.set_title("Resumen de todos los runs comparables")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()  # primero arriba

    # Anotar valores
    for y, p, l in zip(y_pos, peaks, lasts):
        ax.text(p + 0.005, y - bar_height/2, f"{p:.3f}",
                va="center", fontsize=8)
        ax.text(l + 0.005, y + bar_height/2, f"{l:.3f}",
                va="center", fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(OUT / "08_tabla_resumen_barras.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Generación de figuras del TFM — almendros_fl")
    print("=" * 70)
    print(f"  Proyecto: {PROJECT}")
    print(f"  Resultados: {RESULTS}")
    print(f"  Output:     {OUT}")
    print()

    if not RESULTS.exists():
        print(f"ERROR: no existe {RESULTS}")
        sys.exit(1)

    fig_01_curvas_resnet18()
    fig_02_curvas_mobilenetv2_F1()
    fig_03_curvas_mobilenetv2_F2()
    fig_04_comparativa_arch()
    fig_05_brecha_clientes()
    fig_06_fase1_vs_fase2()
    fig_07_overconfidence()
    fig_08_tabla_resumen()

    print()
    print("=" * 70)
    print("  ✅ FIGURAS GENERADAS")
    print("=" * 70)
    print(f"  Las PNGs están en: {OUT}/")
    for png in sorted(OUT.glob("*.png")):
        size_kb = png.stat().st_size / 1024
        print(f"    {png.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
