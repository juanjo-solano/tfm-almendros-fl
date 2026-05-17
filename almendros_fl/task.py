"""almendros_fl: definiciones de modelo, datos y entrenamiento/test.

Arquitectura: MobileNetV2 pre-entrenada en ImageNet + cabeza Dense.
Adaptada del notebook de referencia PRUEBA_CON_FOTOS_NUBLADAS.

Soporta entrenamiento en 2 fases:
  - Fase 1: base (features) congelada, solo se entrena la cabeza (~80K params).
  - Fase 2: descongela las últimas N capas con parámetros entrenables
    (fine-tuning), exactamente igual que en el notebook de referencia.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ───────────────────────────────────────────────────────────
# Configuración global
# ───────────────────────────────────────────────────────────
CONTEXTS = ["manana", "tarde", "nublado", "otros_moviles"]

CLASS_NAMES = ["FOTOS_ALMENDRO_SANO", "FOTOS_MALVAS",
               "FOTOS_ORUGAS_BLANCAS", "FOTOS_VALLICO"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 128
import os

_env_root = os.environ.get("ALMENDROS_DATA_ROOT")
if _env_root:
    DATA_ROOT = Path(_env_root).expanduser().resolve()
else:
    _candidates = [
        Path(__file__).parent.parent / "data",
        Path.cwd() / "data",
    ]
    DATA_ROOT = next((p for p in _candidates if p.exists()),
                     _candidates[0]).resolve()


# ───────────────────────────────────────────────────────────
# Modelo: MobileNetV2 + cabeza Dense (Transfer Learning)
# ───────────────────────────────────────────────────────────
class Net(nn.Module):
    """MobileNetV2 pre-entrenada + cabeza Dense para 4 clases.

    Args:
        num_classes: número de clases de salida (4 en este proyecto).
        phase: 1 = base congelada, 2 = fine-tuning de últimas capas.
        unfreeze_last_n_layers: cuántas capas con parámetros se descongelan
            en fase 2. Por defecto 30 (igual que en la referencia en Keras).

    Estructura:
      Input (3, 128, 128)
        ↓
      MobileNetV2 features  (frozen en fase 1, ~30 capas trainables en fase 2)
        ↓
      AdaptiveAvgPool2d(1)  (equivalente a GlobalAveragePooling2D)
        ↓
      Flatten → Dropout(0.3) → Linear(1280, 64) → ReLU
        ↓
      Dropout(0.3) → Linear(64, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 phase: int = 1,
                 unfreeze_last_n_layers: int = 30):
        super().__init__()

        # Backbone pre-entrenado en ImageNet
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Cabeza Dense (igual estructura que la referencia):
        #   Dropout → Dense(64, ReLU) → Dropout → Dense(num_classes)
        # MobileNetV2 termina con 1280 channels.
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        self.phase = phase
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self._configure_phase()

    def _configure_phase(self) -> None:
        """Configura qué capas son entrenables según la fase actual."""
        # Por defecto, congelar TODA la base
        for param in self.features.parameters():
            param.requires_grad = False

        if self.phase == 2:
            # Aplanar las capas de features que tienen parámetros entrenables
            # (Conv2d, BatchNorm2d, etc.). Saltamos contenedores Sequential.
            trainable_layers = []
            for module in self.features.modules():
                # Solo capas terminales (sin hijos), con parámetros propios
                has_children = any(True for _ in module.children())
                has_own_params = any(True for _ in module.parameters(recurse=False))
                if not has_children and has_own_params:
                    trainable_layers.append(module)

            # Descongelar las últimas N
            n_to_unfreeze = min(self.unfreeze_last_n_layers, len(trainable_layers))
            for layer in trainable_layers[-n_to_unfreeze:]:
                for param in layer.parameters(recurse=False):
                    param.requires_grad = True

        # Cabeza siempre entrenable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_trainable_params(self) -> Tuple[int, int]:
        """Devuelve (entrenables, totales) para verificar fase aplicada."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# ───────────────────────────────────────────────────────────
# Helper: identificar capas BatchNorm para FedBN
# ───────────────────────────────────────────────────────────
def is_bn_key(key: str) -> bool:
    """True si la clave del state_dict pertenece a una capa BatchNorm.

    En MobileNetV2 (torchvision), las capas BN se identifican por el sufijo
    de cada Conv-BN-ReLU dentro de los bloques InvertedResidual:
      - features.X.Y.1.weight / bias / running_mean / running_var
        (donde el .1. dentro del Conv2dNormActivation es la BN)
      - O claves que contengan 'BatchNorm'

    Para más robustez, comprobamos la presencia de las estadísticas BN
    típicas: 'running_mean' y 'running_var' implican que es BN.
    """
    # Las claves de BN siempre tienen running_mean o running_var como sibling
    # en el state_dict. Aquí detectamos por nombre de capa.
    # Con MobileNetV2 de torchvision, las BN están en patrones como:
    #   features.0.1.{weight,bias,running_mean,running_var,num_batches_tracked}
    # donde el segundo índice (.1.) es la BatchNorm dentro de Conv2dNormActivation.
    #
    # Estrategia: si la clave termina en "running_mean", "running_var" o
    # "num_batches_tracked" → es BN. Para weight/bias necesitamos ver si el
    # módulo padre es BatchNorm. Eso lo haremos en el nivel de
    # _filter_bn_keys() inspeccionando el modelo.
    return (key.endswith("running_mean")
            or key.endswith("running_var")
            or key.endswith("num_batches_tracked"))


# ───────────────────────────────────────────────────────────
# Transforms: con augmentation similar al notebook de referencia
# ───────────────────────────────────────────────────────────
def _train_transform():
    """Augmentation siguiendo el notebook de referencia (RandomFlip
    horizontal+vertical, RandomRotation 0.15, RandomZoom 0.15) +
    normalización estándar de ImageNet (requerida por MobileNetV2)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _eval_transform():
    """Sin augmentation, solo normalización."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ───────────────────────────────────────────────────────────
# Wrapper para aplicar transforms distintos sobre el mismo dataset base
# ───────────────────────────────────────────────────────────
class TransformedSubset(torch.utils.data.Dataset):
    """Envuelve un Subset y aplica un transform al vuelo.

    Necesario porque ImageFolder almacena UN único transform, pero tras
    random_split queremos aplicar train_transform a una mitad y eval_transform
    a la otra. La solución canónica es cargar el dataset base SIN transform
    y dejar que este wrapper aplique el que toque en cada __getitem__.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # El dataset base devuelve la PIL Image sin transformar
        # (ImageFolder con transform=None entrega PIL.Image directamente)
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# ───────────────────────────────────────────────────────────
# Carga de datos por partición (un contexto = un cliente)
# ───────────────────────────────────────────────────────────
def load_data(partition_id: int, batch_size: int = 16
              ) -> Tuple[DataLoader, DataLoader]:
    """Carga el contexto del partition_id y devuelve (train_loader, val_loader).

    Una sola instancia de ImageFolder (sin transform), split 80/20 con
    seed=42, y wrappers TransformedSubset que aplican train_transform a
    la partición de entrenamiento y eval_transform a la de validación.
    """
    if partition_id >= len(CONTEXTS):
        raise ValueError(
            f"partition_id={partition_id} fuera de rango. "
            f"Tenemos {len(CONTEXTS)} contextos: {CONTEXTS}"
        )

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"No existe DATA_ROOT={DATA_ROOT}. "
            f"Define ALMENDROS_DATA_ROOT con la ruta absoluta a tu data/."
        )

    context_name = CONTEXTS[partition_id]
    context_dir = DATA_ROOT / context_name

    if not context_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta del contexto {context_name}: {context_dir}"
        )

    # Dataset base SIN transform: ImageFolder entrega PIL.Image en bruto
    base_dataset = ImageFolder(root=str(context_dir), transform=None)

    n_total = len(base_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        base_dataset, [n_train, n_val], generator=generator
    )

    # Cada Subset recibe su propio transform vía wrapper
    train_ds = TransformedSubset(train_subset, _train_transform())
    val_ds = TransformedSubset(val_subset, _eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    return train_loader, val_loader


def load_centralized_test(batch_size: int = 16) -> DataLoader:
    """Test centralizado: combina los 20% val de los 4 contextos.

    Usa la misma seed=42 y la misma fracción 80/20 que load_data,
    garantizando que se evalúa exactamente sobre las mismas imágenes
    que cada cliente usa para validación local.
    """
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"No existe DATA_ROOT={DATA_ROOT}. "
            f"Define la variable de entorno ALMENDROS_DATA_ROOT con la ruta "
            f"absoluta a tu carpeta data/."
        )

    val_datasets = []
    missing = []
    for context_name in CONTEXTS:
        ctx_dir = DATA_ROOT / context_name
        if not ctx_dir.exists():
            missing.append(str(ctx_dir))
            continue

        base_dataset = ImageFolder(root=str(ctx_dir), transform=None)
        n_total = len(base_dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        generator = torch.Generator().manual_seed(42)
        _, val_subset = random_split(
            base_dataset, [n_train, n_val], generator=generator
        )
        val_datasets.append(TransformedSubset(val_subset, _eval_transform()))

    if not val_datasets:
        raise FileNotFoundError(
            f"No se encontró ningún contexto en DATA_ROOT={DATA_ROOT}. "
            f"Faltan: {missing}"
        )

    combined = torch.utils.data.ConcatDataset(val_datasets)
    return DataLoader(combined, batch_size=batch_size,
                      shuffle=False, num_workers=0)


# ───────────────────────────────────────────────────────────
# Funciones de entrenamiento y test locales
# ───────────────────────────────────────────────────────────
def train(net: nn.Module, trainloader: DataLoader, epochs: int,
          lr: float, device: torch.device) -> float:
    """Entrenamiento estándar (FedAvg). El optimizador SOLO actualiza los
    parámetros con requires_grad=True (respeta la fase del modelo)."""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    # Adam funciona mejor que SGD para Transfer Learning con LR pequeños
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    last_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        last_loss = running_loss / max(1, n_batches)

    return last_loss


def test(net: nn.Module, testloader: DataLoader,
         device: torch.device) -> Tuple[float, float]:
    """Evalúa la red y devuelve (loss, accuracy)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def train_proximal(
    net: nn.Module, trainloader: DataLoader, epochs: int,
    lr: float, device: torch.device,
    proximal_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> float:
    """Entrena con loss = CrossEntropy + (mu/2)·||w - w_global||² (FedProx).

    Solo aplica el término proximal sobre los parámetros entrenables
    (los frozen no contribuyen, no tendría sentido)."""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if global_params is not None:
        global_params = [p.to(device) for p in global_params]

    last_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Término proximal solo sobre parámetros entrenables
            if proximal_mu > 0 and global_params is not None:
                prox_term = 0.0
                for local_p, global_p in zip(net.parameters(), global_params):
                    if local_p.requires_grad:
                        prox_term = prox_term + ((local_p - global_p) ** 2).sum()
                loss = loss + (proximal_mu / 2) * prox_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        last_loss = running_loss / max(1, n_batches)

    return last_loss


def set_seed(seed: int) -> None:
    """Fija seeds para reproducibilidad."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
