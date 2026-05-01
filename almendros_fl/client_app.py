"""almendros_fl: ClientApp con soporte para FedAvg, FedProx y FedBN.

Adaptado a MobileNetV2 con entrenamiento en 2 fases (frozen + fine-tuning).
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from almendros_fl.task import (
    Net, load_data, set_seed, test, train_proximal, is_bn_key,
)


_train_mods = []
print("[ClientApp] DP server-side (sin mods de cliente). MobileNetV2 + 2 fases.")

app = ClientApp()


# ───────────────────────────────────────────────────────────
# Helpers FedBN — ahora con detección robusta para MobileNetV2
# ───────────────────────────────────────────────────────────
def _filter_bn_keys(model: nn.Module, state_dict: dict) -> dict:
    """Devuelve solo las claves del state_dict que pertenecen a capas BN.

    Inspecciona el modelo para identificar qué módulos son BatchNorm,
    así cubrimos correctamente MobileNetV2 (donde las BN no tienen 'bn'
    en el nombre, sino que están dentro de Conv2dNormActivation).
    """
    bn_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_module_names.add(name)

    bn_state = {}
    for k, v in state_dict.items():
        # k tiene forma "features.0.1.weight" → módulo padre es "features.0.1"
        module_path = ".".join(k.split(".")[:-1])
        if module_path in bn_module_names:
            bn_state[k] = v
    return bn_state


def _apply_local_bn(model: nn.Module, local_bn: dict) -> None:
    current = model.state_dict()
    for k, v in local_bn.items():
        if k in current:
            current[k] = v
    model.load_state_dict(current)


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
@app.train(mods=_train_mods)
def train_fn(msg: Message, context: Context) -> Message:
    # ── Config recibida del servidor ──
    lr: float = msg.content["config"]["lr"]
    proximal_mu: float = float(msg.content["config"].get("proximal_mu", 0.0))
    fedbn: bool = bool(msg.content["config"].get("fedbn", False))

    # ── Config local del cliente ──
    local_epochs: int = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch-size"]
    seed: int = int(context.run_config.get("seed", 42))
    phase: int = int(context.run_config.get("phase", 1))
    unfreeze_n: int = int(context.run_config.get("unfreeze-last-n-layers", 30))
    partition_id = int(context.node_config["partition-id"])

    set_seed(seed + partition_id)

    if fedbn:
        print(f"[Cliente {partition_id}] Train FedBN | fase={phase} | lr={lr}")
    elif proximal_mu > 0:
        print(f"[Cliente {partition_id}] Train FedProx mu={proximal_mu} | "
              f"fase={phase} | lr={lr}")
    else:
        print(f"[Cliente {partition_id}] Train FedAvg | fase={phase} | lr={lr}")

    # ── Construir modelo según la fase ──
    model = Net(phase=phase, unfreeze_last_n_layers=unfreeze_n)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Re-aplicar la configuración de fase tras cargar pesos
    # (load_state_dict no toca requires_grad, pero por si acaso)
    model._configure_phase()

    # Logging de parámetros entrenables (solo en cliente 0 para no spamear)
    if partition_id == 0:
        trainable, total = model.count_trainable_params()
        pct = 100.0 * trainable / total if total > 0 else 0.0
        print(f"[Cliente 0] Params entrenables: {trainable:,} / {total:,} "
              f"({pct:.1f}%)")

    # ── FedBN: restaurar BN locales ──
    if fedbn:
        bn_state_record = context.state.array_records.get("bn_state")
        if bn_state_record is not None:
            local_bn = bn_state_record.to_torch_state_dict()
            _apply_local_bn(model, local_bn)
            print(f"[Cliente {partition_id}] BN locales restauradas "
                  f"({len(local_bn)} tensores)")
        else:
            print(f"[Cliente {partition_id}] Primera ronda: sin BN locales aún")

    # ── Guardar pesos globales para el término proximal ──
    global_params = [p.detach().clone() for p in model.parameters()]

    # ── Datos ──
    trainloader, _ = load_data(partition_id, batch_size=batch_size)

    # ── Entrenar ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loss = train_proximal(
        model, trainloader, epochs=local_epochs, lr=lr,
        device=device, proximal_mu=proximal_mu, global_params=global_params,
    )

    # ── FedBN: guardar BN locales ──
    if fedbn:
        bn_only = _filter_bn_keys(model, model.state_dict())
        if bn_only:
            context.state.array_records["bn_state"] = ArrayRecord(bn_only)
            print(f"[Cliente {partition_id}] BN locales guardadas "
                  f"({len(bn_only)} tensores)")
        else:
            print(f"[Cliente {partition_id}] FedBN sin efecto: no hay BN "
                  f"entrenables en fase {phase}")

    # ── Respuesta ──
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "_partition_id": float(partition_id),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# ───────────────────────────────────────────────────────────
# Evaluate
# ───────────────────────────────────────────────────────────
@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    fedbn: bool = bool(msg.content["config"].get("fedbn", False))
    batch_size: int = context.run_config["batch-size"]
    phase: int = int(context.run_config.get("phase", 1))
    unfreeze_n: int = int(context.run_config.get("unfreeze-last-n-layers", 30))
    partition_id = int(context.node_config["partition-id"])
    print(f"[Cliente {partition_id}] Evaluación local")

    model = Net(phase=phase, unfreeze_last_n_layers=unfreeze_n)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    if fedbn:
        bn_state_record = context.state.array_records.get("bn_state")
        if bn_state_record is not None:
            local_bn = bn_state_record.to_torch_state_dict()
            _apply_local_bn(model, local_bn)

    _, valloader = load_data(partition_id, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loss, val_acc = test(model, valloader, device)

    metrics = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "num-examples": len(valloader.dataset),
        "_partition_id": float(partition_id),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
