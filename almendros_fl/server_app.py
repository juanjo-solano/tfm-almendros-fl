"""almendros_fl: ServerApp con soporte para múltiples estrategias federadas."""

import time
from typing import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    FedAvg, FedAvgM, FedProx,
    DifferentialPrivacyServerSideFixedClipping,
)

from almendros_fl.logger import ExperimentLogger
from almendros_fl.task import CONTEXTS, Net, load_centralized_test, set_seed, test

app = ServerApp()


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────
def _extract_client_metrics(reply: Message) -> dict:
    """Extrae las métricas del cliente sin asumir bajo qué clave vienen.

    Recorre los MetricRecord del RecordDict y devuelve un dict plano. Si hay
    varios MetricRecord (raro), los fusiona; el último gana en colisiones.
    """
    out: dict = {}
    try:
        # API moderna: iterar sobre los MetricRecord del RecordDict
        for mr in reply.content.metric_records.values():
            out.update(dict(mr))
    except AttributeError:
        # Fallback: intentar como diccionario clásico
        try:
            mr = reply.content.get("metrics", {})
            out.update(dict(mr))
        except Exception:
            pass
    return out


# ───────────────────────────────────────────────────────────
# Mixin de logging compartido entre estrategias
# ───────────────────────────────────────────────────────────
class LoggingMixin:
    """Añade logging unificado (CSV + W&B) + tracking del best model
    para early stopping a cualquier estrategia FL."""

    def _attach_logger(self, logger: ExperimentLogger):
        self._exp_logger = logger
        self._round_start: float | None = None
        self._train_per_client: dict[int, dict] = {}
        self._eval_per_client: dict[int, dict] = {}
        self._global_metrics: dict[str, float] = {}
        self._round_train_loss_agg: float | None = None
        # ── Early stopping (selección del best model) ──
        self._best_accuracy: float = -1.0
        self._best_round: int = 0
        self._best_state_dict: dict | None = None
        self._patience_counter: int = 0
        self._early_stop_triggered: bool = False

    def configure_train(
        self, server_round: int, arrays: ArrayRecord,
        config: ConfigRecord, grid: Grid,
    ) -> Iterable[Message]:
        self._round_start = time.time()
        self._train_per_client.clear()
        self._eval_per_client.clear()
        self._global_metrics = {}
        self._round_train_loss_agg = None
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # La firma oficial es tupla (ArrayRecord, MetricRecord)
        arrays, metrics = super().aggregate_train(server_round, replies)

        # Guardar métricas por cliente leyendo directamente los replies
        for reply in replies:
            cm = _extract_client_metrics(reply)
            pid = int(cm.get("_partition_id", -1))
            if pid >= 0:
                self._train_per_client[pid] = {
                    "train_loss": float(cm.get("train_loss", 0.0)),
                    "num_examples": int(cm.get("num-examples", 0)),
                }

        # Train loss agregado (media ponderada)
        total_ex = sum(c["num_examples"] for c in self._train_per_client.values())
        if total_ex > 0:
            self._round_train_loss_agg = sum(
                c["train_loss"] * c["num_examples"]
                for c in self._train_per_client.values()
            ) / total_ex

        return arrays, metrics

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message],
    ) -> MetricRecord | None:
        agg_metrics = super().aggregate_evaluate(server_round, replies)

        for reply in replies:
            cm = _extract_client_metrics(reply)
            pid = int(cm.get("_partition_id", -1))
            if pid >= 0:
                self._eval_per_client[pid] = {
                    "val_loss": float(cm.get("val_loss", 0.0)),
                    "val_accuracy": float(cm.get("val_accuracy", 0.0)),
                    "num_examples": int(cm.get("num-examples", 0)),
                }

        # NO hacemos flush aquí: aún no tenemos global_loss/accuracy de esta ronda.
        # El flush lo hace record_global_eval al final del evaluate_fn central.
        return agg_metrics

    # ── Volcado al logger ────────────────────────────────
    def _flush_client_logs(self, server_round: int):
        all_pids = set(self._train_per_client) | set(self._eval_per_client)
        for pid in sorted(all_pids):
            train = self._train_per_client.get(pid, {})
            evald = self._eval_per_client.get(pid, {})
            ctx = CONTEXTS[pid] if 0 <= pid < len(CONTEXTS) else f"pid_{pid}"
            self._exp_logger.log_client_round(
                round_num=server_round,
                partition_id=pid,
                context=ctx,
                train_loss=train.get("train_loss"),
                val_loss=evald.get("val_loss"),
                val_accuracy=evald.get("val_accuracy"),
                num_examples=evald.get("num_examples") or train.get("num_examples"),
            )

    def _flush_server_round(self, server_round: int):
        total_ex = sum(c["num_examples"] for c in self._eval_per_client.values())
        val_loss_agg = None
        val_acc_agg = None
        if total_ex > 0:
            val_loss_agg = sum(
                c["val_loss"] * c["num_examples"]
                for c in self._eval_per_client.values()
            ) / total_ex
            val_acc_agg = sum(
                c["val_accuracy"] * c["num_examples"]
                for c in self._eval_per_client.values()
            ) / total_ex

        round_time = (
            time.time() - self._round_start if self._round_start else None
        )

        self._exp_logger.log_server_round(
            round_num=server_round,
            global_loss=self._global_metrics.get("global_loss"),
            global_accuracy=self._global_metrics.get("global_accuracy"),
            train_loss_agg=self._round_train_loss_agg,
            val_loss_agg=val_loss_agg,
            val_accuracy_agg=val_acc_agg,
            round_time_s=round_time,
        )

    def record_global_eval(self, server_round: int, loss: float, accuracy: float,
                           current_state_dict: dict | None = None,
                           patience: int = 3):
        """Llamado por el evaluate_fn central. Hace flush de la ronda completa
        y actualiza el tracking del best model para early stopping."""
        self._global_metrics = {"global_loss": loss, "global_accuracy": accuracy}

        # ── Early stopping: tracking del best model ──
        if server_round >= 1 and current_state_dict is not None:
            if accuracy > self._best_accuracy:
                self._best_accuracy = accuracy
                self._best_round = server_round
                # Copia profunda del state_dict (clone tensores)
                self._best_state_dict = {
                    k: v.detach().clone() for k, v in current_state_dict.items()
                }
                self._patience_counter = 0
                print(f"[EarlyStop] Nuevo mejor accuracy={accuracy:.4f} "
                      f"en ronda {server_round}")
            else:
                self._patience_counter += 1
                print(f"[EarlyStop] Sin mejora ({self._patience_counter}/{patience}). "
                      f"Best={self._best_accuracy:.4f} en ronda {self._best_round}")

                if (self._patience_counter >= patience
                        and not self._early_stop_triggered):
                    self._early_stop_triggered = True
                    print(f"[EarlyStop] TRIGGER en ronda {server_round}: "
                          f"paciencia agotada. El best model queda fijado "
                          f"(ronda {self._best_round}, "
                          f"acc={self._best_accuracy:.4f}). "
                          f"Las rondas restantes se ejecutaran pero no se usaran.")

        # Flush solo si la ronda ya pasó por evaluate clientes (server_round >= 1).
        # La ronda 0 es la evaluación inicial sin clientes, no hay nada que loggear.
        if server_round >= 1:
            self._flush_client_logs(server_round)
            self._flush_server_round(server_round)


# ───────────────────────────────────────────────────────────
# Versiones con logging
# ───────────────────────────────────────────────────────────
class LoggedFedAvg(LoggingMixin, FedAvg):
    pass


class LoggedFedProx(LoggingMixin, FedProx):
    pass


class LoggedFedAvgM(LoggingMixin, FedAvgM):
    pass


# ───────────────────────────────────────────────────────────
# Factory
# ───────────────────────────────────────────────────────────
def build_strategy(name: str, run_config: dict, **common):
    name = name.lower()
    if name == "fedavg":
        return LoggedFedAvg(**common)
    if name == "fedprox":
        mu = float(run_config.get("proximal-mu", 0.01))
        return LoggedFedProx(proximal_mu=mu, **common)
    if name == "fedavgm":
        momentum = float(run_config.get("server-momentum", 0.9))
        return LoggedFedAvgM(server_momentum=momentum, **common)
    if name == "fedbn":
        # FedBN se implementa en el cliente; el servidor agrega como FedAvg.
        return LoggedFedAvg(**common)
    raise ValueError(
        f"Estrategia desconocida: '{name}'. "
        f"Valores válidos: 'fedavg', 'fedprox', 'fedavgm', 'fedbn'."
    )


# ───────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────
@app.main()
def main(grid: Grid, context: Context):
    cfg = context.run_config
    strategy_name = str(cfg.get("strategy", "fedavg"))
    num_rounds = int(cfg["num-server-rounds"])
    phase = int(cfg.get("phase", 1))
    if phase == 2:
        lr = float(cfg.get("learning-rate-phase2", cfg.get("learning-rate", 1e-5)))
    else:
        lr = float(cfg.get("learning-rate-phase1", cfg.get("learning-rate", 1e-3)))
    print(f"[Servidor] phase={phase} | learning_rate={lr}")
    seed = int(cfg.get("seed", 42))

    set_seed(seed)

    # Detectar si DP está activado para añadirlo al run_id
    dp_suffix = "_DP" if cfg.get("dp-enabled", False) else ""
    logger = ExperimentLogger(
        strategy_name=f"{strategy_name.capitalize()}{dp_suffix}",
        wandb_project=str(cfg.get("wandb-project", "almendros-fl")),
        wandb_mode=str(cfg.get("wandb-mode", "offline")),
        results_dir=str(cfg.get("results-dir", "results")),
        run_config=dict(cfg),
    )

    phase_cfg = int(cfg.get("phase", 1))
    unfreeze_n = int(cfg.get("unfreeze-last-n-layers", 30))
    net = Net(phase=phase_cfg, unfreeze_last_n_layers=unfreeze_n)
    print(f"[Servidor] Net(phase={phase_cfg}, unfreeze_last_n_layers={unfreeze_n})")

    # Soporte de init-from: cargar pesos de un checkpoint anterior
    # (típicamente para que Fase 2 arranque desde el final_model.pt de Fase 1)
    init_from = cfg.get("init-from", "")
    if init_from:
        from pathlib import Path
        init_path = Path(init_from).expanduser()
        if init_path.exists():
            try:
                state = torch.load(str(init_path), map_location="cpu", weights_only=True)
            except Exception:
                state = torch.load(str(init_path), map_location="cpu")
            net.load_state_dict(state)
            print(f"[Servidor] Pesos iniciales cargados desde: {init_path}")
        else:
            print(f"[Servidor] AVISO: init-from='{init_path}' no existe, se ignora.")
    arrays = ArrayRecord(net.state_dict())

    common_kwargs = dict(
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
    )
    strategy = build_strategy(strategy_name, cfg, **common_kwargs)
    strategy._attach_logger(logger)

    # Guardar referencia a la estrategia "logueable" antes de envolverla.
    # Esto nos permite seguir invocando _attach_logger / record_global_eval
    # aunque la envolvamos con DP, porque el wrapper no expone esos métodos.
    logged_strategy = strategy

    # ── Envolver con DP central si está activado ──
    dp_enabled = bool(cfg.get("dp-enabled", False))
    if dp_enabled:
        dp_clipping_norm = float(cfg.get("dp-clipping-norm", 10.0))
        dp_noise_multiplier = float(cfg.get("dp-noise-multiplier", 1.0))
        num_sampled = max(1, int(4 * float(cfg["fraction-train"])))

        print(
            f"[Servidor] Central DP activado: "
            f"clipping_norm={dp_clipping_norm}, "
            f"noise_multiplier={dp_noise_multiplier}, "
            f"num_sampled_clients={num_sampled}"
        )
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy=strategy,
            noise_multiplier=dp_noise_multiplier,
            clipping_norm=dp_clipping_norm,
            num_sampled_clients=num_sampled,
        )

    print(f"[Servidor] Estrategia activa: {strategy.__class__.__name__}")

    # ── Construir configs (train y evaluate) ──
    train_config_dict: dict = {"lr": lr}
    evaluate_config_dict: dict = {}

    if strategy_name == "fedprox":
        mu = float(cfg.get("proximal-mu", 0.01))
        train_config_dict["proximal_mu"] = mu
        print(f"[Servidor] proximal_mu = {mu}")
    elif strategy_name == "fedavgm":
        print(f"[Servidor] server_momentum = {cfg.get('server-momentum', 0.9)}")
    elif strategy_name == "fedbn":
        train_config_dict["fedbn"] = True
        evaluate_config_dict["fedbn"] = True
        print(f"[Servidor] Modo FedBN: BatchNorm local en cada cliente")

    # ── Early stopping (selección del best model) ──
    patience = int(cfg.get("early-stopping-patience", 3))
    print(f"[Servidor] Early stopping activado: patience={patience}, "
          f"monitor=global_accuracy")

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net(phase=phase_cfg, unfreeze_last_n_layers=unfreeze_n)
        state_dict = arrays.to_torch_state_dict()
        model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loader = load_centralized_test(batch_size=32)
        loss, acc = test(model, test_loader, device)
        print(
            f"[Servidor] Ronda {server_round} | "
            f"global_loss={loss:.4f} | global_accuracy={acc:.4f}"
        )
        logged_strategy.record_global_eval(
            server_round, loss, acc,
            current_state_dict=state_dict, patience=patience,
        )
        return MetricRecord({"global_loss": loss, "global_accuracy": acc})

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_config_dict),
        evaluate_config=ConfigRecord(evaluate_config_dict),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # ── Guardar modelo final (último estado tras todas las rondas) ──
    final_path = logger.run_dir / "final_model.pt"
    torch.save(result.arrays.to_torch_state_dict(), final_path)
    print(f"[Servidor] Modelo final guardado en: {final_path}")

    # ── Guardar best model (early stopping: ronda con mejor global_accuracy) ──
    if logged_strategy._best_state_dict is not None:
        best_path = logger.run_dir / "best_model.pt"
        torch.save(logged_strategy._best_state_dict, best_path)
        print(
            f"[Servidor] Best model guardado en: {best_path} "
            f"(ronda {logged_strategy._best_round}, "
            f"acc={logged_strategy._best_accuracy:.4f})"
        )

        # Anotar info de early stopping en config.txt
        with open(logger.run_dir / "config.txt", "a") as f:
            f.write(f"best_round={logged_strategy._best_round}\n")
            f.write(f"best_accuracy={logged_strategy._best_accuracy:.6f}\n")
            f.write(f"early_stop_triggered={logged_strategy._early_stop_triggered}\n")
            f.write(f"early_stop_patience={patience}\n")
    else:
        print("[Servidor] No se guardo best_model (no hubo evaluaciones).")

    logger.finish()
