#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_confusion_matrices.py

Genera la figura con las matrices de confusión del mejor modelo
(MobileNetV2 Fase 2 + FedProx) evaluado en el conjunto de validación
de cada uno de los cuatro clientes.

Ejecución (desde el directorio del proyecto, con el venv activado):
    cd ~/TFM/Flower_ejemplo/almendros/almendros_fl
    python3 gen_confusion_matrices.py

Salida:
    figs/matrices_confusion_fedprox_f2.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from almendros_fl.task import (
    CLASS_NAMES,
    CONTEXTS,
    Net,
    load_data,
)

MODEL_PATH = (
    PROJECT
    / "results"
    / "_docker_results_phase2_20260429-1406"
    / "Fedprox_20260429-094241-767"
    / "final_model.pt"
)
OUT = PROJECT / "figs"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHORT_NAMES = ["Sano", "Malvas", "Orugas", "Vallico"]
CONTEXT_LABELS = ["Mañana", "Tarde", "Nublado", "Otros\nmóviles"]


def load_model() -> Net:
    net = Net(phase=2)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    return net


def predict(net: Net, loader) -> tuple[np.ndarray, np.ndarray]:
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = net(images)
            preds = outputs.argmax(dim=1).cpu()
            all_labels.append(labels)
            all_preds.append(preds)
    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_cm(ax, cm: np.ndarray, title: str, class_names: list[str]) -> None:
    n = len(class_names)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Clase predicha", fontsize=9)
    ax.set_ylabel("Clase real", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val_norm = cm_norm[i, j]
            val_abs = cm[i, j]
            color = "white" if val_norm > thresh else "black"
            ax.text(
                j, i,
                f"{val_abs}\n({val_norm:.0%})",
                ha="center", va="center",
                color=color, fontsize=7.5,
            )

    return im


def main() -> None:
    print(f"Modelo: {MODEL_PATH}")
    print(f"Dispositivo: {DEVICE}")

    net = load_model()
    print("Modelo cargado correctamente.\n")

    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for partition_id in range(4):
        context = CONTEXTS[partition_id]
        print(f"  Cliente {partition_id} ({context}) …", end=" ", flush=True)

        _, val_loader = load_data(partition_id, batch_size=32)
        y_true, y_pred = predict(net, val_loader)
        cm = confusion_matrix(y_true, y_pred, n_classes)

        acc = (y_true == y_pred).mean()
        title = f"Cliente {partition_id}: {CONTEXT_LABELS[partition_id]} (acc={acc:.1%})"

        im = plot_cm(axes[partition_id], cm, title, SHORT_NAMES)
        print(f"acc={acc:.3f}")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Fracción por fila (recall)")

    fig.suptitle(
        "Matrices de confusión por cliente\n"
        "MobileNetV2 Fase 2 + FedProx (μ=0.001) — conjunto de validación local",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    out_path = OUT / "matrices_confusion_fedprox_f2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅  Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
