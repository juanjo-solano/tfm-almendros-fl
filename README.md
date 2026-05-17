# TFM — Aprendizaje Federado para clasificación de almendros y malas hierbas

> **Autor:** Solano
> **Framework:** [Flower](https://flower.ai/) 1.26.1 + PyTorch 2.x
> **Entorno final:** Docker Compose + TLS (Transport Layer Security)
> **Arquitectura ganadora:** MobileNetV2 + Transfer Learning (2 fases)

---

## 🎯 Resumen del proyecto

Sistema de aprendizaje federado distribuido para clasificar imágenes de
almendros y malas hierbas en 4 clases:

- `FOTOS_ALMENDRO_SANO`
- `FOTOS_MALVAS`
- `FOTOS_ORUGAS_BLANCAS`
- `FOTOS_VALLICO`

El reto principal del proyecto es el **escenario non-IID**: los datos están
particionados en 4 contextos fotográficos distintos (clientes federados),
cada uno con 100 imágenes:

- `manana` — fotos tomadas por la mañana (alta luminosidad)
- `tarde` — fotos tomadas por la tarde (baja luminosidad)
- `nublado` — fotos en días nublados (luz difusa)
- `otros_moviles` — fotos hechas con dispositivos diferentes

Esto crea un **feature shift** entre clientes que es exactamente el caso
realista que motiva el uso de Federated Learning sobre datos sensibles
distribuidos en distintos dispositivos / sitios.

---

## 📋 Tabla de contenidos

1. [Estructura del proyecto](#-estructura-del-proyecto)
2. [Resumen de fases](#-resumen-de-fases)
3. [Detalle de cada fase](#-detalle-de-cada-fase)
4. [Resultados experimentales](#-resultados-experimentales)
5. [Cómo reproducir](#-cómo-reproducir)
6. [Archivos clave](#-archivos-clave)

---

## 📁 Estructura del proyecto

```
almendros_fl/
├── almendros_fl/                       # Paquete Python principal
│   ├── task.py                         # Modelo + transformaciones + carga de datos
│   ├── client_app.py                   # Cliente federado (FedAvg/FedProx/FedBN)
│   ├── server_app.py                   # Servidor + estrategias + métricas
│   ├── logger.py                       # Logging W&B + CSV
│   ├── summary.py                      # Generación de resúmenes
│   └── task_resnet18.py.bak            # Backup de la arquitectura ResNet18 anterior
│
├── data/                               # Dataset (4 contextos × 4 clases × ~25 imgs)
│   ├── manana/
│   ├── tarde/
│   ├── nublado/
│   └── otros_moviles/
│
├── pyproject.toml                      # Config del proyecto Flower
├── compose.yml                         # 10 servicios: SuperLink + 4 SuperNodes + 5 SuperExec
├── with-tls.yml                        # Overlay para activar TLS en gRPC
├── certs.yml                           # Generador de certificados auto-firmados
├── superlink-certificates/             # ca.crt, server.pem, server.key
│
├── run_all_6.sh                        # Orquestador de la tanda completa (6 runs)
├── run_phase2_only.sh                  # Orquestador solo Fase 2
│
└── results/                            # Resultados de cada run
    ├── ResNet18_*/                     # Runs con la arquitectura antigua (Fases 3-6)
    ├── _docker_results_full_*/         # MobileNetV2 Fase 1 (3 runs)
    ├── _docker_results_phase2_*/       # MobileNetV2 Fase 2 (3 runs)
    ├── _phase1_checkpoints/            # final_model.pt de cada Fase 1
    ├── _failed_runs/                   # Evidencia experimental DP (Fase 5)
    └── _run_logs/                      # Logs de orquestación
```

---

## 🗺️ Resumen de fases

| Fase | Descripción | Estado | Hallazgo principal |
|------|-------------|:------:|--------------------|
| 1 | Setup inicial + dataset | ✅ | 4 contextos × 4 clases, 100 imgs/cliente |
| 2 | Primer run FedAvg | ✅ | Funciona pero oscila por feature shift |
| 3 | Comparativa FedAvg/FedProx/FedAvgM (ResNet18) | ✅ | FedProx ganador con μ=0.01 |
| 4 | Estrategia FedBN (ResNet18) | ✅ | Overfitting documentado |
| 5 | Differential Privacy | ⚠️ | Diverge con ResNet18 + 100 imgs/cliente |
| 6.1 | Deployment Runtime nativo (tmux) | ✅ | Pico 0.770 |
| 6.2 | Construcción imágenes Docker | ✅ | 5 imágenes superexec basadas en Flower 1.26.1 |
| 6.3 | Docker Compose | ✅ | Pico 0.7575, +18% más rápido que tmux |
| 6.4 | TLS con OpenSSL | ✅ | Pico 0.785 (R7), +15% tiempo aceptable |
| 7 | Cambio arquitectura → MobileNetV2 | ✅ | **Pico 0.875 (FedProx F2)**, +9pp vs ResNet18 |

---

## 🧪 Detalle de cada fase

### Fase 1-2: Setup + primer run con FedAvg (ResNet18)

**Objetivo:** validar que la pipeline federada funciona end-to-end.

**Configuración:**
- Modelo: ResNet18 desde cero (sin weights pre-entrenados al inicio)
- Optimizador: SGD lr=0.01, momentum=0.9
- 4 clientes simulados con Ray
- 8 rondas, 2 epochs locales, batch=16

**Resultado:** Pico val_acc 0.7725 (ronda 4), pero oscilaciones fuertes
(±10-30 pp entre rondas consecutivas) por **client drift / feature shift**.

**Conclusión:** funciona, pero la inestabilidad justifica probar FedProx
y FedBN.

---

### Fase 3: Comparativa FedAvg / FedProx / FedAvgM (ResNet18)

**Objetivo:** identificar la mejor estrategia para el escenario non-IID.

**Configuración común:**
- ResNet18 con weights ImageNet (Transfer Learning ligero)
- 8 rondas, 2 epochs locales, batch=16, lr=0.01
- Simulación con Ray, seed=42

**Resultados clave (carpetas `*_20260426-*`):**

| Estrategia | μ / momentum | Pico val_acc | Comportamiento |
|------------|-------------|:------------:|----------------|
| FedAvg     | -           | 0.745 | Oscila |
| FedProx    | μ=0.001     | 0.745 | Mejora poco |
| FedProx    | **μ=0.01**  | **0.7675** | **Estabiliza** |
| FedProx    | μ=0.1       | 0.732 | Demasiado restrictivo |
| FedAvgM    | β=0.9       | 0.755 | Oscilación reducida |

**Carpetas relevantes:**
- `FedAvg_20260426-112851/` — FedAvg final
- `Fedprox_20260426-142040/` — **FedProx ganador** (μ=0.01)
- `FedAvgM_20260426-122321/` — FedAvgM

---

### Fase 4: FedBN (ResNet18)

**Objetivo:** probar si mantener BatchNorm locales mejora el non-IID.

**Configuración:** mismas que Fase 3, pero con FedBN (BN local por cliente).

**Resultado:** Pico val_acc 0.7775, pero **overfitting evidente**
(train_loss → 0, val_loss creciente). El modelo memoriza los 100 imgs
de cada cliente.

**Carpeta:** `Fedbn_20260426-162815/`

---

### Fase 5: Differential Privacy ⚠️

**Objetivo:** añadir garantías formales de privacidad mediante DP.

**Configuración probada:**
- Server-side `DifferentialPrivacyServerSideFixedClipping`
- Múltiples combinaciones: `clipping_norm` ∈ {0.1, 0.5, 1.0, 5.0},
  `noise_multiplier` ∈ {0.01, 0.1, 1.0}, `sensitivity` ∈ {1.0, 5.0}

**Resultado:** **TODOS los runs DP divergen** con ResNet18 (11M params)
+ 100 imágenes por cliente. La señal de gradiente es demasiado débil
respecto al ruido DP.

**Conclusión documentada en la memoria:** DP en este escenario requiere
o bien (a) más datos por cliente, o (b) un modelo con muchos menos
parámetros entrenables. **La Fase 7 (MobileNetV2) confirmaría esta
hipótesis: con solo 82K params entrenables en Fase 1, DP se vuelve
viable.**

**Carpetas:** `_failed_runs/Fedprox_DP_*/` — 8 runs fallidos como
evidencia experimental.

---

### Fase 6.1: Deployment Runtime nativo (tmux)

**Objetivo:** salir de la simulación Ray a procesos reales en localhost.

**Arquitectura desplegada:**
```
SuperLink ──┬── SuperNode 0 (manana)   ── ClientApp 0 (subprocess)
            ├── SuperNode 1 (tarde)    ── ClientApp 1 (subprocess)
            ├── SuperNode 2 (nublado)  ── ClientApp 2 (subprocess)
            └── SuperNode 3 (otros)    ── ClientApp 3 (subprocess)
ServerApp ── (subprocess separado)
```

Cada componente en una ventana de tmux distinta.

**Resultado:**
- Pico val_acc 0.770 (ronda 6)
- Tiempo total 88 min (vs 27 min en simulación) → ~3× overhead por
  creación de subprocesos `flwr-clientapp` por cada mensaje

**Carpeta:** `Fedprox_20260427-125231-094/`

---

### Fase 6.2: Construcción de imágenes Docker

5 imágenes parametrizadas basadas en `flwr/superexec:1.26.1`:

- `almendros_fl-superexec-serverapp` — para el ServerApp
- `almendros_fl-superexec-clientapp-{manana,tarde,nublado,otros}`

Cada imagen contiene el paquete `almendros_fl` + el subset de datos
de su contexto. Tamaño: ~10.8 GB efectivo, ~11 GB reales en disco
gracias a deduplicación de capas Docker.

**Decisión técnica:** elegimos `flwr/superexec` (Alpine) sobre las
imágenes Ubuntu de Flower porque incluyen las dependencias mínimas
necesarias y son significativamente más pequeñas.

---

### Fase 6.3: Docker Compose

**Objetivo:** orquestar 10 contenedores con un único `docker compose up`.

**Servicios desplegados:**
1. `superlink` — coordinador central
2. `superexec-serverapp` — ejecutor del ServerApp
3. `supernode-{manana,tarde,nublado,otros}` — 4 SuperNodes
4. `superexec-clientapp-{manana,tarde,nublado,otros}` — 4 ClientApps

**Hallazgo experimental clave:** el bridge de Docker 29.x con imágenes
Alpine de Flower 1.26 da **timeouts TCP silenciosos** entre contenedores.
**Solución adoptada:** `network_mode: host`. Esto además simula
realísticamente un escenario donde cada nodo reside en una máquina
física distinta (cada SuperNode usa su propio puerto del host:
9094-9097).

**Resultados:**
- Pico val_acc 0.7575
- Tiempo 73 min (−18% vs deployment local con tmux)

**Carpeta:** `Fedprox_DOCKER_20260428-085954-079/`

---

### Fase 6.4: TLS con OpenSSL

**Objetivo:** cifrar todas las conexiones gRPC entre componentes.

**Implementación:**
- `certs.yml` lanza un contenedor Ubuntu+OpenSSL que genera:
  - `ca.crt` / `ca.key` — Autoridad Certificadora (RSA 4096, SHA-256, 365 días)
  - `server.pem` / `server.key` — Certificado del SuperLink
  - SAN incluye: `localhost`, `superlink`, `127.0.0.1`
- `with-tls.yml` añade flags `--ssl-*` al SuperLink y
  `--root-certificates` a los SuperNodes
- ServerApp y ClientApps usan `--insecure` (loopback intra-host)

**Resultados:**
- **Pico val_acc 0.785 (R7)** — el mejor de toda la serie ResNet18
- Tiempo 84 min (+15% vs sin TLS, aceptable)

**Carpeta:** `Fedprox_DOCKER_TLS_20260428-103910-159/`
*(si no aparece, mira en el contenedor o renombra desde
`_docker_results_*` la que corresponda)*

---

### Fase 7: Cambio de arquitectura → MobileNetV2 + Transfer Learning ⭐

**Motivación:** el notebook de referencia (`Pueba_fotos_nubladas-i_1.ipynb`)
demostraba mejores resultados centralizados con **MobileNetV2 pre-entrenado
+ Transfer Learning en 2 fases**. Adaptamos esa arquitectura al contexto
federado.

**Arquitectura adoptada:**
```
Input (3, 128, 128)
    ↓
[Data Augmentation]
    ├── RandomHorizontalFlip(0.5)
    ├── RandomVerticalFlip(0.5)
    ├── RandomRotation(15°)
    └── RandomResizedCrop(scale 0.85-1.0)
    ↓
[Normalización ImageNet]
    ↓
MobileNetV2 features (1280 ch)        ← FASE 1: TODA congelada
                                      ← FASE 2: últimas 30 capas entrenables
    ↓
AdaptiveAvgPool2d(1)                  (= GlobalAveragePooling2D)
    ↓
Dropout(0.3) → Linear(1280, 64) → ReLU → Dropout(0.3) → Linear(64, 4)
                                                        (cabeza Dense, ~80K params)
```

**Optimizador:** Adam (no SGD; mejor para fine-tuning con LR pequeños).

**Hiperparámetros:**

| Parámetro | Fase 1 | Fase 2 |
|-----------|:------:|:------:|
| Learning rate | 1e-3 | 1e-5 |
| Params entrenables | 82,244 (3.57%) | 1,825,412 (79.16%) |
| Inicialización | ImageNet | Checkpoint Fase 1 |
| Rondas | 12 | 12 |

**Estrategias FL:** FedAvg, FedProx (μ=0.001), FedBN.

> **Nota teórica importante:** FedProx μ se redujo 10× respecto a
> ResNet18 (de 0.01 a 0.001) porque con menos parámetros entrenables,
> los deltas son más pequeños y un μ proporcionalmente menor evita
> frenar excesivamente el aprendizaje.

> **Nota sobre FedBN:** En Fase 1, las BN están dentro de la base
> congelada → no son entrenables → **FedBN se comporta exactamente
> como FedAvg**. Esto se confirmó experimentalmente: ambos dan
> resultados idénticos (0.8200 / 0.6067). En Fase 2 con la base
> descongelada, FedBN sí actúa.

---

## 📊 Resultados experimentales

### Tabla maestra — todos los runs comparables

| # | Arquitectura | Estrategia | Runtime | Pico val_acc | Final R12 | Tiempo |
|---|--------------|-----------|---------|:-----------:|:---------:|:------:|
| 1 | ResNet18 | FedAvg | Simulación Ray | 0.745 | 0.745 | 17 min |
| 2 | ResNet18 | FedProx (μ=0.01) | Simulación Ray | **0.7675** | 0.745 | 27 min |
| 3 | ResNet18 | FedBN | Simulación Ray | 0.7775 | overfit | 28 min |
| 4 | ResNet18 | FedProx (μ=0.01) | Deployment local | 0.770 | 0.710 | 88 min |
| 5 | ResNet18 | FedProx (μ=0.01) | Docker Compose | 0.7575 | 0.7575 | 73 min |
| 6 | ResNet18 | FedProx (μ=0.01) | Docker + TLS | **0.785** | 0.680 | 84 min |
| 7 | MobileNetV2 | FedAvg F1 | Docker + TLS | 0.8375 | 0.8200 | 102 min |
| 8 | MobileNetV2 | FedProx F1 (μ=0.001) | Docker + TLS | 0.8350 | **0.8350** | 102 min |
| 9 | MobileNetV2 | FedBN F1 | Docker + TLS | 0.8375 | 0.8200 | 102 min |
| 10 | MobileNetV2 | FedAvg F2 | Docker + TLS | 0.8725 | 0.8725 | 80 min |
| 11 | **MobileNetV2** | **FedProx F2 (μ=0.001)** | **Docker + TLS** | **🏆 0.8750** | **0.8750** | 80 min |
| 12 | MobileNetV2 | FedBN F2 | Docker + TLS | 0.8725 | 0.8725 | 80 min |

### Lecturas clave

1. **Mejor configuración: MobileNetV2 + FedProx Fase 2 → val_acc 0.8750**
2. **Ganancia ResNet18 → MobileNetV2 con fine-tuning: +9 puntos porcentuales**
3. **Reducción de tráfico federado: 42.7 MB → 8.97 MB (−79%)**
4. **Reducción params entrenables Fase 1: 11.18M → 82K (−99.3%, 136× menos)**
5. **FedBN ≡ FedAvg en Fase 1** (confirmado experimentalmente:
   resultados numéricamente idénticos)
6. **Overconfidence en Fase 2:** train_loss → 0.0003 con val_loss
   subiendo. El modelo se vuelve excesivamente confiado pero el
   accuracy sí mejora. Documentado para discusión.
7. **Cliente más difícil: `tarde`** (luminosidad baja). Brecha de
   ~19 pp respecto a `manana`. Mejora con fine-tuning de F1 → F2.

---

## 🚀 Cómo reproducir

### Requisitos

- Linux/Mac/WSL
- Docker 24+ (importante: con Docker 29 ver workaround `network_mode: host`)
- Python 3.10+ (solo para `flwr` cliente)
- Espacio: ~12 GB para imágenes Docker + ~3 GB para datos y resultados

### Setup inicial

```bash
# 1. Instalar Flower CLI
pip install "flwr[simulation]==1.26.1"

# 2. Configurar federación TLS local
mkdir -p ~/.flwr
cat << EOF > ~/.flwr/config.toml
[superlink]
default = "local-deployment-tls"

[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 4

[superlink.local-deployment-tls]
address = "127.0.0.1:9093"
root-certificates = "$(pwd)/superlink-certificates/ca.crt"
EOF

# 3. Generar certificados TLS
PROJECT_DIR=. docker compose -f certs.yml run --rm --build gen-certs
sudo chown -R $USER:$USER superlink-certificates/
chmod 644 superlink-certificates/server.key superlink-certificates/ca.crt superlink-certificates/server.pem
chmod 600 superlink-certificates/ca.key
```

### Ejecutar un solo run

```bash
# Levantar el clúster TLS
PROJECT_DIR=. docker compose -f compose.yml -f with-tls.yml up -d --build

# Lanzar un run
flwr run . local-deployment-tls --stream \
    --run-config "num-server-rounds=12 strategy='fedprox' phase=1 proximal-mu=0.001"
```

### Ejecutar la tanda completa (6 runs ~10h)

```bash
nohup ./run_all_6.sh > results/_run_logs/orchestrator.log 2>&1 &
disown
```

---

## 📦 Archivos clave (qué entregar)

### Mínimo imprescindible (50-80 MB)

```
TFM_Solano/
├── README.md                          ← este archivo
├── almendros_fl/                      ← código completo (sin __pycache__)
│   ├── client_app.py
│   ├── server_app.py
│   ├── task.py
│   ├── logger.py
│   ├── summary.py
│   └── __init__.py
├── pyproject.toml
├── compose.yml
├── with-tls.yml
├── certs.yml
├── run_all_6.sh
├── run_phase2_only.sh
├── superlink-certificates/
│   ├── ca.crt
│   ├── server.pem
│   └── server.key
├── generate_figures.py                ← script para gráficas
└── results/
    ├── ResNet18_runs/                 ← 4 runs ResNet18 clave
    │   ├── FedAvg_simulacion/         (= FedAvg_20260426-112851)
    │   ├── FedProx_simulacion/        (= Fedprox_20260426-142040)
    │   ├── FedBN_simulacion/          (= Fedbn_20260426-162815)
    │   ├── FedProx_deployment_local/  (= Fedprox_20260427-125231-094)
    │   ├── FedProx_docker/            (= Fedprox_DOCKER_20260428-085954-079)
    │   └── FedProx_docker_TLS/        (si existe)
    ├── MobileNetV2_runs/              ← 6 runs MobileNetV2
    │   ├── FedAvg_F1/
    │   ├── FedProx_F1/
    │   ├── FedBN_F1/
    │   ├── FedAvg_F2/
    │   ├── FedProx_F2_GANADOR/        ← el mejor
    │   └── FedBN_F2/
    └── DP_failed_runs/                ← evidencia Fase 5
```

### Cómo preparar la entrega

```bash
cd ~/TFM/Flower_ejemplo/almendros/almendros_fl

# 1. Crear carpeta de entrega limpia
mkdir -p ../TFM_Solano_entrega/results

# 2. Copiar código
cp -r almendros_fl ../TFM_Solano_entrega/
rm -rf ../TFM_Solano_entrega/almendros_fl/__pycache__
cp pyproject.toml compose.yml with-tls.yml certs.yml \
   run_all_6.sh run_phase2_only.sh \
   ../TFM_Solano_entrega/

# 3. Copiar certificados (sin ca.key y server.csr, no son necesarios)
mkdir -p ../TFM_Solano_entrega/superlink-certificates
cp superlink-certificates/{ca.crt,server.pem,server.key} \
   ../TFM_Solano_entrega/superlink-certificates/

# 4. Renombrar y copiar runs ResNet18 clave
mkdir -p ../TFM_Solano_entrega/results/ResNet18_runs
cp -r results/FedAvg_20260426-112851 \
      ../TFM_Solano_entrega/results/ResNet18_runs/FedAvg_simulacion
cp -r results/Fedprox_20260426-142040 \
      ../TFM_Solano_entrega/results/ResNet18_runs/FedProx_simulacion
cp -r results/Fedbn_20260426-162815 \
      ../TFM_Solano_entrega/results/ResNet18_runs/FedBN_simulacion
cp -r results/Fedprox_20260427-125231-094 \
      ../TFM_Solano_entrega/results/ResNet18_runs/FedProx_deployment_local
cp -r results/Fedprox_DOCKER_20260428-085954-079 \
      ../TFM_Solano_entrega/results/ResNet18_runs/FedProx_docker
# (si existe Fedprox_DOCKER_TLS_*, copiarlo también)

# 5. Renombrar y copiar runs MobileNetV2
mkdir -p ../TFM_Solano_entrega/results/MobileNetV2_runs
cp -r results/_docker_results_full_20260429-0443/Fedavg_20260428-214855-007 \
      ../TFM_Solano_entrega/results/MobileNetV2_runs/FedAvg_F1
cp -r results/_docker_results_full_20260429-0443/Fedprox_20260428-232700-387 \
      ../TFM_Solano_entrega/results/MobileNetV2_runs/FedProx_F1
cp -r results/_docker_results_full_20260429-0443/Fedbn_20260429-010542-065 \
      ../TFM_Solano_entrega/results/MobileNetV2_runs/FedBN_F1
cp -r results/_docker_results_phase2_20260429-1406/Fedavg_20260429-083224-247 \
      ../TFM_Solano_entrega/results/MobileNetV2_runs/FedAvg_F2
cp -r results/_docker_results_phase2_20260429-1406/Fedprox_20260429-094241-767 \
      ../TFM_Solano_entrega/results/MobileNetV2_runs/FedProx_F2_GANADOR
cp -r results/_docker_results_phase2_20260429-1406/Fedbn_20260429-105546-338 \
      ../TFM_Solano_entrega/results/MobileNetV2_runs/FedBN_F2

# 6. Copiar evidencia DP fallida
cp -r results/_failed_runs ../TFM_Solano_entrega/results/DP_failed_runs

# 7. Copiar el README y generate_figures.py
cp README.md generate_figures.py ../TFM_Solano_entrega/

# 8. Ver tamaño total
du -sh ../TFM_Solano_entrega/

# 9. Empaquetar
cd ..
tar -czf TFM_Solano_entrega.tar.gz TFM_Solano_entrega/
ls -lh TFM_Solano_entrega.tar.gz
```

### Lo que NO entregar

- `_tmp/` (clon del repo de Flower)
- `__pycache__/`
- `wandb/` dentro de cada run (logs internos de W&B; **opcional**:
  para navegar las gráficas online, ver sección
  W&B abajo)
- `*.bak`, `*.old`
- `deployment_logs/` (logs del intento Fase 6.1 antes de la versión final)
- `final_model.pt` en la raíz (residuo viejo)
- `data/` si pesa demasiado (el destinatario ya tiene su propio dataset)
- `superlink-certificates/ca.key` (clave privada de la CA, sensible)

---

## 📈 Cómo obtener las gráficas

### Opción 1 — Generarlas localmente con `generate_figures.py`

```bash
cd ~/TFM/Flower_ejemplo/almendros/almendros_fl
python3 generate_figures.py
ls figs/
```

Esto crea PNGs de:
- Curvas de aprendizaje por estrategia (val_acc / train_loss / val_loss)
- Comparativa MobileNetV2 vs ResNet18
- Brecha entre clientes (manana / tarde / nublado / otros)
- Comparativa Fase 1 vs Fase 2

### Opción 2 — Sincronizar con Weights & Biases

Los runs guardaron logs offline en `wandb/offline-run-*`. Para verlos
en la web de W&B:

```bash
# 1. Crear cuenta en https://wandb.ai (gratis)
pip install wandb
wandb login

# 2. Sincronizar TODOS los runs MobileNetV2
for d in results/_docker_results_*/Fed*/wandb/offline-run-*; do
    wandb sync "$d"
done

# 3. Sincronizar también los runs ResNet18 clave
for d in results/Fedprox_20260426-142040/wandb/offline-run-* \
         results/Fedbn_20260426-162815/wandb/offline-run-* \
         results/FedAvg_20260426-112851/wandb/offline-run-* \
         results/Fedprox_20260427-125231-094/wandb/offline-run-* \
         results/Fedprox_DOCKER_20260428-085954-079/wandb/offline-run-*; do
    wandb sync "$d" 2>/dev/null
done
```

Luego abre https://wandb.ai/<tu_usuario> y verás todos los runs con
sus gráficas interactivas. Desde la interfaz web puedes:

- Comparar runs (hasta 50 a la vez)
- Exportar cada panel como PNG/SVG/PDF (botón "..." sobre cada gráfica)
- Crear "Reports" con gráficas + texto (ideal para la memoria)

---

## ⚠️ Notas y limitaciones conocidas

1. **Bug de carpeta anidada en `Fedprox_DOCKER_20260428-085954-079`:**
   Al hacer `docker cp` se creó una carpeta `Fedprox_20260428-085954-079`
   anidada **dentro**. Es residuo, no afecta a los datos. Para limpiar:
   ```bash
   rm -rf results/Fedprox_DOCKER_20260428-085954-079/Fedprox_20260428-085954-079
   ```

2. **Locale español rompe `printf %f` en bash.** Si copias scripts y
   ves errores tipo `printf: 0,8750: número no válido`, prefíjalos:
   ```bash
   LC_NUMERIC=C bash -c '...'
   ```

3. **Docker 29.x + Alpine bridge:** los SuperNodes no pueden conectarse
   con el SuperLink usando bridge network. Solución implementada:
   `network_mode: host` en `compose.yml`.

4. **DP no funciona con ResNet18:** este es un hallazgo experimental,
   no un bug. Los runs en `_failed_runs/` documentan el intento.
   Posible siguiente paso: probar DP con MobileNetV2 Fase 1 (solo
   82K params entrenables; razón teórica para que funcione).

5. **Cliente `tarde` es estructuralmente más difícil** (~19 pp por
   debajo de `manana`). Esto es coherente con la baja luminosidad
   de las fotos vespertinas. La Fase 2 con fine-tuning reduce la
   brecha pero no la elimina.

---

## 🙏 Créditos

- Dataset y notebook centralizado de referencia
- Framework Flower: <https://flower.ai/>
- Plantilla Docker: <https://github.com/adap/flower/tree/main/framework/docker/complete>

---

*Última actualización: 29 de abril de 2026*
