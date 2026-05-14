"""almendros_fl: utilidades de logging unificadas (CSV + W&B)."""

import csv
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _get_results_dir(results_dir: str) -> Path:
    """Devuelve la ruta absoluta a la carpeta de resultados."""
    env_root = os.environ.get("ALMENDROS_RESULTS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.cwd() / results_dir).resolve()


class ExperimentLogger:
    """Logger unificado para CSV + W&B."""

    def __init__(
        self,
        strategy_name: str,
        wandb_project: str,
        wandb_mode: str,
        results_dir: str,
        run_config: dict[str, Any],
    ):
        self.strategy_name = strategy_name
        self.start_time = time.time()

        # Carpeta única para este experimento
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        seed = run_config.get("seed", "NA")
        self.run_name = f"{strategy_name}_seed{seed}_{timestamp}"
        base_dir = _get_results_dir(results_dir)
        self.run_dir = base_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Verificar que la carpeta es realmente escribible
        test_file = self.run_dir / ".write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
        except OSError as e:
            raise RuntimeError(
                f"[Logger] No se puede escribir en {self.run_dir}: {e}. "
                f"¿Espacio en disco lleno? ¿Permisos?"
            )

        print(f"[Logger] Resultados en: {self.run_dir}")

        # Inicializar CSVs
        self.server_csv_path = self.run_dir / "server_metrics.csv"
        self.client_csv_path = self.run_dir / "client_metrics.csv"
        self._init_csvs()

        # Guardar configuración
        self._save_config(run_config)

        # W&B con manejo robusto de errores
        self.wandb_run = None
        if _WANDB_AVAILABLE and wandb_mode != "disabled":
            self._init_wandb(wandb_project, wandb_mode, run_config)
        else:
            print("[Logger] W&B desactivado, solo CSV.")

    def _init_wandb(self, project: str, mode: str, run_config: dict) -> None:
        """Intenta inicializar W&B mostrando la excepción real si falla."""
        for attempt_mode in (mode, "offline"):
            try:
                self.wandb_run = wandb.init(
                    project=project,
                    name=self.run_name,
                    config={"strategy": self.strategy_name, **run_config},
                    mode=attempt_mode,
                    dir=str(self.run_dir),
                )
                print(f"[Logger] W&B inicializado en modo '{attempt_mode}'")
                return
            except Exception as e:
                print(f"[Logger] W&B falló en modo '{attempt_mode}'.")
                print(f"[Logger] Excepción: {type(e).__name__}: {e}")
                if attempt_mode == mode and attempt_mode != "offline":
                    print(f"[Logger] Intentando modo offline como fallback...")
                else:
                    print(f"[Logger] Traceback completo:")
                    traceback.print_exc()
                    print(f"[Logger] Continuando solo con CSV.")
                    self.wandb_run = None
                    return

    def _init_csvs(self) -> None:
        if not self.server_csv_path.exists():
            with open(self.server_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round", "global_loss", "global_accuracy",
                    "train_loss_agg", "val_loss_agg", "val_accuracy_agg",
                    "round_time_s", "elapsed_s",
                ])
        if not self.client_csv_path.exists():
            with open(self.client_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round", "partition_id", "context",
                    "train_loss", "val_loss", "val_accuracy",
                    "num_examples",
                ])

    def _save_config(self, run_config: dict[str, Any]) -> None:
        with open(self.run_dir / "config.txt", "w") as f:
            f.write(f"strategy={self.strategy_name}\n")
            f.write(f"timestamp={datetime.now().isoformat()}\n")
            for k, v in run_config.items():
                f.write(f"{k}={v}\n")

    def log_server_round(
        self,
        round_num: int,
        global_loss: float | None = None,
        global_accuracy: float | None = None,
        train_loss_agg: float | None = None,
        val_loss_agg: float | None = None,
        val_accuracy_agg: float | None = None,
        round_time_s: float | None = None,
    ) -> None:
        elapsed = time.time() - self.start_time

        try:
            with open(self.server_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    round_num,
                    _fmt(global_loss), _fmt(global_accuracy),
                    _fmt(train_loss_agg), _fmt(val_loss_agg),
                    _fmt(val_accuracy_agg),
                    _fmt(round_time_s), f"{elapsed:.2f}",
                ])
        except OSError as e:
            print(f"[Logger] Error escribiendo server_metrics.csv: {e}")

        if self.wandb_run is not None:
            payload = {"round": round_num, "elapsed_s": elapsed}
            for name, val in [
                ("global_loss", global_loss),
                ("global_accuracy", global_accuracy),
                ("train_loss_agg", train_loss_agg),
                ("val_loss_agg", val_loss_agg),
                ("val_accuracy_agg", val_accuracy_agg),
                ("round_time_s", round_time_s),
            ]:
                if val is not None:
                    payload[name] = val
            try:
                self.wandb_run.log(payload, step=round_num)
            except Exception as e:
                print(f"[Logger] Error logueando a W&B: {e}")

    def log_client_round(
        self,
        round_num: int,
        partition_id: int,
        context: str,
        train_loss: float | None = None,
        val_loss: float | None = None,
        val_accuracy: float | None = None,
        num_examples: int | None = None,
    ) -> None:
        try:
            with open(self.client_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    round_num, partition_id, context,
                    _fmt(train_loss), _fmt(val_loss), _fmt(val_accuracy),
                    num_examples if num_examples is not None else "",
                ])
        except OSError as e:
            print(f"[Logger] Error escribiendo client_metrics.csv: {e}")

        if self.wandb_run is not None:
            payload = {f"client_{partition_id}_{context}/round": round_num}
            for name, val in [
                ("train_loss", train_loss),
                ("val_loss", val_loss),
                ("val_accuracy", val_accuracy),
            ]:
                if val is not None:
                    payload[f"client_{partition_id}_{context}/{name}"] = val
            try:
                self.wandb_run.log(payload, step=round_num)
            except Exception as e:
                print(f"[Logger] Error logueando a W&B: {e}")

    def finish(self) -> None:
        total_time = time.time() - self.start_time
        try:
            with open(self.run_dir / "config.txt", "a") as f:
                f.write(f"total_time_s={total_time:.2f}\n")
        except OSError as e:
            print(f"[Logger] Error escribiendo config.txt: {e}")
        print(f"[Logger] Experimento finalizado en {total_time:.2f}s")
        print(f"[Logger] Resultados: {self.run_dir}")
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"[Logger] Error cerrando W&B: {e}")


def _fmt(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)