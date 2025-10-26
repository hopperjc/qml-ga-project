# QML-GA Project — Variational Quantum Classifiers otimizados por Algoritmos Genéticos

Repositório para experimentos com **Variational Quantum Classifiers (VQC)** usando **Algoritmos Genéticos (AG)** e otimizadores por gradiente (Adam/Nesterov) com **PennyLane**.  
A pipeline varre automaticamente **datasets**, **feature maps** (Amplitude/ZZ), **ansätze** (`ansatz_1 … ansatz_6`) e **hiperparâmetros**, registrando **resultados reprodutíveis**.

---

## Sumário
- [Estrutura do projeto](#estrutura-do-projeto)
- [Ambiente de desenvolvimento](#ambiente-de-desenvolvimento)
  - [Pré-requisitos](#pré-requisitos)
  - [Criar o ambiente com Poetry](#criar-o-ambiente-com-poetry)
  - [VS Code](#vs-code)
- [Configurações (YAML)](#configurações-yaml)
  - [Datasets](#datasets)
  - [Feature Maps](#feature-maps)
  - [Ansätze](#ansätze)
  - [Hipergrids](#hipergrids)
  - [Devices e Optimizers (opcionais)](#devices-e-optimizers-opcionais)
- [Dados & pré-processamento](#dados--pré-processamento)
- [Como rodar os experimentos](#como-rodar-os-experimentos)
  - [Dry-run (validação/listagem)](#1-dry-run-validaçãolistagem)
  - [Treinando de fato](#2-treinando-de-fato)
  - [Flags úteis](#flags-úteis)
- [Saídas e avaliação](#saídas-e-avaliação)

---

## Estrutura do projeto

qml-ga-project/
├─ .venv/ # ambiente virtual do Poetry (criado localmente)
├─ configs/
│ ├─ ansatz/ # ansätze e profundidades (e.g., ansatz_1_d6.yaml, ansatz_1_d15.yaml, …)
│ ├─ datasets/ # definição dos datasets (CSV processado, coluna alvo, etc.)
│ ├─ devices/ # (opcional) presets de devices; o sweep infere wires automaticamente
│ ├─ feature_maps/ # amplitude.yaml, zz.yaml
│ ├─ hypergrids/ # grades de hiperparâmetros (ga.yaml, classical.yaml)
│ └─ optimizers/ # (opcional) configs fixas por otimizador para execuções manuais
│
├─ data/
│ ├─ raw/ # dados brutos (UCI etc.)
│ └─ processed/ # CSVs prontos para treino (normalizados)
│
├─ logs/ # logs de execução (opcional)
├─ notebooks/ # pré-processamento e análises exploratórias
├─ reports/ # relatórios por combinação (summary.json, folds.csv, config.yaml) + index.csv
├─ runs/ # artefatos completos de cada execução (um diretório por run)
│
├─ src/
│ └─ qml_ga/
│ ├─ ansatz/ # ansatz_1 … ansatz_6 + base.py (shape canônico dos pesos)
│ ├─ circuits/ # vqc.py (QNode: bias é somado fora do QNode)
│ ├─ data/ # datamodule.py (carrega CSV + valida wires)
│ ├─ experiments/ # kfold_train.py (treino/validação K-Fold)
│ ├─ feature_maps/ # amplitude.py, zz.py
│ ├─ metrics/ # métricas de classificação
│ ├─ optimizers/ # ga.py (PyGAD 2.x/3.x), classical.py (Adam/Nesterov)
│ ├─ sweep/ # cli.py (entrypoint: qmlga-sweep)
│ ├─ utils/ # io.py (YAML, diretórios, etc.)
│ └─ cli.py # entrypoint: qmlga
│
├─ .gitignore
├─ poetry.lock
├─ pyproject.toml
└─ README.md


---

## Ambiente de desenvolvimento

### Pré-requisitos
- **Python 3.12 x64**
- **Poetry**
- Git e VS Code (recomendados)

### Criar o ambiente

```bash
# criar venv local
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# selecionar o Python 3.12
poetry env use 3.12

# instalar dependências
poetry install
```

---

Configurações (YAML)
Datasets

Arquivos em configs/datasets/*.yaml:

dataset:
  name: banknote_authentication
  path: data/processed/banknote_authentication/dataset_normalized.csv
  target: class


Os rótulos são mapeados para {-1, +1} automaticamente quando necessário.

Feature Maps

Arquivos em configs/feature_maps/*.yaml.

amplitude.yaml

feature_map:
  type: amplitude


zz.yaml

feature_map:
  type: zz
  params:
    alpha: 1.0


Regras de wires (derivadas automaticamente no sweep):

Amplitude: wires = ceil(log2(n_features))

ZZ: wires = n_features

Ansätze

Arquivos como configs/ansatz/ansatz_<i>_d<depth>.yaml:

ansatz:
  type: ansatz_1
  params:
    depth: 15


Parâmetros por wire (P) por ansatz:

ansatz_1, ansatz_2, ansatz_6 → P = 3

ansatz_3, ansatz_4, ansatz_5 → P = 5

Shape do tensor de pesos: (depth, n_wires, P).

Hipergrids

configs/hypergrids/ga.yaml

ga:
  population_size: [50, 100, 150]
  num_generations: [50, 100, 200]
  selection_type: ["tournament", "sss"]
  crossover_type: ["single_point", "two_points_crossover"]
  mutation_type: ["random", "adaptive_mutation"]
  mutation_percent_genes: [10]
  elitism: [2]
  init_range_low: [-3.14]
  init_range_high: [3.14]


configs/hypergrids/classical.yaml

classical:
  adam:
    lr: [0.05]
    epochs: [50, 200]
    batch_size: [16]
  nesterov:
    lr: [0.05]
    epochs: [50, 200]
    batch_size: [16]

Devices e Optimizers (opcionais)

configs/devices/: presets para execuções manuais.

configs/optimizers/: valores fixos por otimizador para execuções fora do sweep.

Observação: no sweep, os wires são inferidos automaticamente de dataset + feature_map.

Dados & pré-processamento

Coloque os dados brutos em data/raw/<dataset>/....

Gere um único CSV normalizado por dataset em data/processed/<dataset>/dataset_normalized.csv.

Crie o YAML correspondente em configs/datasets/ apontando para o CSV e a coluna target.

Como rodar os experimentos
1) Dry-run (validação/listagem)
poetry run qmlga-sweep --dry_run --include_feature_maps amplitude --max_wires 12

2) Treinando de fato

Amplitude

poetry run qmlga-sweep --include_feature_maps amplitude --max_wires 12


ZZ

poetry run qmlga-sweep --include_feature_maps zz --max_wires 12

Flags úteis
--trace          # imprime traceback completo em caso de falha
--abort_on_fail  # aborta na primeira falha
--limit N        # executa apenas N combinações (smoke test)

Saídas e avaliação

Para cada combinação (dataset × feature map × ansatz × depth × otimizador), são gerados:

runs/<timestamp>_run/summary.json — artefatos da execução;

reports/<TAG>/summary.json — média/desvio das métricas (K-Fold);

reports/<TAG>/folds.csv — métricas por dobra (accuracy, precision, recall, f1);

reports/<TAG>/config.yaml — configuração exata utilizada;

reports/index.csv — índice global com uma linha por execução.

Formato da TAG:

<dataset>__<feature_map>__<ansatz>_d<depth>__<optimizer>__w<wires>__...


Avaliação:

Use reports/index.csv para tabelas agregadas por combinação.

Use os folds.csv de cada TAG para análises estatísticas (ex.: comparar GA vs Adam/Nesterov para a mesma arquitetura).