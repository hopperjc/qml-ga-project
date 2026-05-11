# QML-GA — Variational Quantum Classifiers otimizados por Algoritmos Genéticos

Repositório de experimentos do mestrado em **Quantum Machine Learning** + **Algoritmos Genéticos** no CIN-UFPE.
Compara AG (via PyGAD) contra otimizadores por gradiente (Adam, Nesterov Momentum) no treinamento de
**Variational Quantum Classifiers (VQCs)** em PennyLane, variando feature maps (Amplitude, ZZ),
seis topologias de ansatz (C1–C6), profundidades (L ∈ {6, 15, 20}) e **regime de ruído quântico**
(amplitude/phase damping parametrizados).

> **Status (2026-05):** infraestrutura SLURM removida; paralelismo agora via
> `concurrent.futures.ProcessPoolExecutor`. Suporte a ruído quântico via `default.mixed` (canais de
> *amplitude* e *phase damping* parametrizados por γ) adicionado. Suíte de testes `pytest`
> cobrindo as invariantes críticas dos canais de ruído.

---

## Sumário

- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [CLIs disponíveis](#clis-disponíveis)
- [Como executar](#como-executar)
- [Suporte a ruído quântico](#suporte-a-ruído-quântico)
- [Testes](#testes)
- [Reprodutibilidade](#reprodutibilidade)

---

## Estrutura do projeto

```
qml-ga-project/
├── src/qml_ga/                  # Pacote Python principal
│   ├── ansatz/                  # 6 topologias de ansatz (ansatz_1..ansatz_6) + dispatcher
│   ├── circuits/vqc.py          # build_vqc — escolhe default.qubit/default.mixed por noise_config
│   ├── data/                    # Carregamento + pré-processamento (K-fold estratificado)
│   ├── experiments/             # kfold_train.py — pipeline experimental
│   ├── feature_maps/            # Amplitude Encoding + ZZ Feature Map
│   ├── noise/                   # Canais de ruído (amplitude/phase damping, depolarizing)
│   │   ├── channels.py          # Primitivos: apply_amplitude_damping, apply_phase_damping
│   │   └── policy.py            # apply_noise_layer + requires_mixed_device (dispatcher por config)
│   ├── optimizers/              # ga.py (PyGAD wrapper), classical.py (Adam/Nesterov via PennyLane)
│   ├── metrics/                 # acurácia, precisão, recall, F1, testes estatísticos
│   ├── sweep/
│   │   ├── cli.py               # CLI principal (qmlga-sweep) — combo generation + sharding manual
│   │   ├── runner.py            # ProcessPoolExecutor — instrumentação tempo+memória automática
│   │   └── best.py              # qmlga-beststats — Friedman/ANOVA/Shapiro
│   ├── single/cli.py            # qmlga-single — 1 experimento isolado
│   └── utils/                   # io.py (YAML, run_id), logger.py (CSV/texto)
│
├── configs/
│   ├── datasets/                # banknote, sonar, wdbc (UCI)
│   ├── feature_maps/
│   │   ├── amplitude.yaml       # Baseline sem ruído
│   │   ├── zz.yaml              # Baseline sem ruído
│   │   └── noisy_examples/      # 6 configs de referência com bloco `noise:`
│   ├── ansatz/                  # 18 YAMLs (ansatz1_d6/d15/d20, ansatz2_*, ..., ansatz6_*)
│   ├── hypergrids/              # ga.yaml + classical.yaml — grades de hiperparâmetros
│   └── devices/                 # wires, shots (opcional)
│
├── tests/                       # pytest — invariantes de ruído + smoke tests
│   ├── conftest.py              # Fixtures sintéticas (sem dependência de data/)
│   ├── test_circuits.py
│   ├── test_feature_maps.py
│   ├── test_noise.py            # γ=0 idempotente, γ=1 colapso (CRÍTICO)
│   └── test_smoke.py            # End-to-end @pytest.mark.smoke
│
├── data/                        # Datasets em CSV (raw/processed/splits)
├── reports/                     # Saídas dos experimentos (summary.json, folds.csv por tag)
├── runs/                        # Snapshots de configuração por execução
└── pyproject.toml               # Dependências (Poetry)
```

> **Nota:** o texto da dissertação (LaTeX ABNT), os PDFs dos artigos de referência, os comentários
> da banca e a apresentação não fazem parte deste repositório público — são mantidos em local
> separado pelo autor.

---

## Instalação

### Pré-requisitos

- **Python 3.12** (constraint em `pyproject.toml: python = ">=3.12,<3.13"`)
- **Poetry** para gestão de dependências

### Setup

```bash
poetry install
```

> **Nota sobre Python 3.13:** se seu Python global é 3.13, o `poetry install` falha porque
> `numpy 1.26.4` não tem wheel pré-compilado pra 3.13. Soluções:
> - Instalar Python 3.12 e apontar `poetry env use python3.12`
> - **OU** afrouxar a constraint em `pyproject.toml` para `python = ">=3.12,<3.14"` e
>   `numpy = ">=2.0"` (numpy 2.x tem wheel pra 3.13; pennylane 0.39 é compatível).

---

## CLIs disponíveis

Após `poetry install`, ficam disponíveis:

| Comando | Função |
|---|---|
| `qmlga-single` | Roda 1 experimento isolado (1 dataset × 1 FM × 1 ansatz × 1 otimizador) |
| `qmlga-sweep` | Varredura paralela de combinações via `ProcessPoolExecutor` |
| `qmlga-beststats` | Análise pós-sweep (Friedman, ANOVA, Shapiro–Wilk) |

---

## Como executar

### Validação rápida (dry-run, sem rodar nada)

```bash
poetry run qmlga-sweep --dry_run --limit 5 --workers 2
```

### Varredura completa (com paralelismo Python)

```bash
poetry run qmlga-sweep --workers 4 --resume
```

- `--workers N`: número de processos paralelos via `ProcessPoolExecutor` (default 1, serial). Em CPU recomenda-se `N = nº cores / 2`.
- `--resume`: pula combinações cujo `reports/{tag}/status.json` já indica `done=True`.

### Sharding manual (opcional, dividir entre máquinas)

```bash
# Máquina 1
poetry run qmlga-sweep --workers 4 --shard_index 0 --shard_total 2 --resume

# Máquina 2
poetry run qmlga-sweep --workers 4 --shard_index 1 --shard_total 2 --resume
```

Cada combinação é roteada para um único shard via `(idx-1) % shard_total == shard_index`.

> **Histórico:** o sharding antes era automático via variáveis `SLURM_*` (cluster CIN-UFPE).
> Após a refatoração pós-banca, toda a infraestrutura SLURM foi removida — sharding agora é
> argumento explícito do CLI.

### Experimento isolado

```bash
poetry run qmlga-single \
    --dataset_yaml configs/datasets/banknote.yaml \
    --feature_map_yaml configs/feature_maps/amplitude.yaml \
    --ansatz_yaml configs/ansatz/ansatz1_d6.yaml \
    --optimizer ga \
    --wires 2
```

### Análise estatística pós-sweep

```bash
poetry run qmlga-beststats --reports_dir reports
```

---

## Suporte a ruído quântico

Adicionar um bloco `noise:` a qualquer arquivo `configs/feature_maps/*.yaml` ativa simulação com
matriz densidade (`default.mixed`) e canais de Kraus realistas:

```yaml
feature_map:
  type: "amplitude"
  params:
    normalize: true
  noise:
    after_feature_map:
      type: amplitude_damping
      gamma: 0.01
    per_ansatz_layer:
      type: amplitude_damping
      gamma: 0.005
```

**Pontos de aplicação configuráveis** (cada um opcional):
- `after_feature_map`: ruído imediatamente após a codificação dos dados
- `per_ansatz_layer`: ruído após cada camada do ansatz (escala com profundidade)

**Tipos de canal suportados** (no módulo [`src/qml_ga/noise/channels.py`](src/qml_ga/noise/channels.py)):
- `amplitude_damping` — relaxação T₁ (perda de excitação)
- `phase_damping` — dephasing T₂ (perda de coerência sem perda de energia)
- `depolarizing` — bonus, útil para comparação

**Backwards-compatibility**: omitir o bloco `noise:` → device volta a `default.qubit`, comportamento
idêntico ao pré-refatoração.

**Configurações de referência prontas** em [`configs/feature_maps/noisy_examples/`](configs/feature_maps/noisy_examples/):
6 YAMLs cobrindo regimes `after_*` e `combined`, com γ ∈ {0.01, 0.05}.

Para rodar uma varredura completa só com configs ruidosas:
```bash
poetry run qmlga-sweep \
    --feature_maps_dir configs/feature_maps/noisy_examples \
    --include_feature_maps amplitude,zz \
    --workers 4 --resume
```


---

## Testes

A suíte cobre invariantes críticas dos canais de ruído + smoke end-to-end:

```bash
# Rápidos (< 30s), sem fold de K-fold
poetry run pytest -m "not smoke"

# End-to-end (1 run por configuração, ~5–10 minutos)
poetry run pytest -m smoke -v
```

**Testes críticos** ([tests/test_noise.py](tests/test_noise.py)):
- `γ=0` em qualquer canal → resultado **numericamente idêntico** ao `default.qubit` puro (≤ 1e-9)
- `γ=1` em amplitude damping → estado colapsa para `|0⟩`, ⟨Z₀⟩ = +1
- `γ=1` em phase damping em `|+⟩` → completa decoerência, ⟨Z₀⟩ = 0
- `γ=0.5` em phase damping em `|0⟩` → estado Z-eigen preservado, ⟨Z₀⟩ = +1

Esses testes são a salvaguarda contra regressões silenciosas no modelo de ruído.

---

## Reprodutibilidade

- **Sementes fixas**: 42 (particionamento K-fold, inicialização de pesos, operadores estocásticos do AG)
- **Snapshots de config**: cada execução grava `runs/{timestamp}_run/config_snapshot.yaml`
- **Status incremental**: `reports/sweep_status.{run_id}.json` permite retomar interrupções via `--resume`
- **Instrumentação automática**: cada `reports/{tag}/summary.json` inclui `wall_time_seconds` e `peak_memory_mb`
  (medidos via `time.perf_counter()` + `tracemalloc`)

Em uma máquina limpa:
```bash
git clone <repo>
cd qml-ga-project
poetry install
poetry run pytest                     # valida ambiente
poetry run qmlga-sweep --dry_run --limit 1   # valida configs
```

---

## Citação

Trabalho prévio do autor publicado em IJCNN 2025:

```bibtex
@inproceedings{costa2025evolutionary,
  title={Evolutionary Weight Optimization for Variational Quantum Classifiers},
  author={Costa, Matheus Hopper Jansen and Neto, Fernando M. de Paula},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  year={2025},
  organization={IEEE}
}
```

---

## Licença

Centro de Informática — Universidade Federal de Pernambuco (CIN-UFPE).
