# Feature maps com ruído (referência)

Esta pasta contém YAMLs de referência para experimentos com ruído quântico.
**Não** são pegos pelo sweep default (`configs/feature_maps/*.yaml`).

## Como usar

Para rodar uma varredura usando estes feature maps:

```
qmlga-sweep \
    --feature_maps_dir configs/feature_maps/noisy_examples \
    --include_feature_maps amplitude,zz \
    --workers 4 --resume
```

## Nomenclatura

`<fm>_<regime>_g<gamma>.yaml`:
- `<fm>`: `amplitude` ou `zz`
- `<regime>`:
  - `after`  → ruído só depois do feature map (preparação imperfeita)
  - `combined` → ruído depois do FM + após cada camada do ansatz (NISQ realista)
- `<gamma>`: γ formatado sem ponto (ex: `g01` = 0.01, `g05` = 0.05, `g005` = 0.005)

## Schema do bloco `noise:`

```yaml
noise:
  after_feature_map:                  # opcional
    type: amplitude_damping | phase_damping | depolarizing
    gamma: <float in [0, 1]>
  per_ansatz_layer:                   # opcional
    type: ...
    gamma: ...
```

Slots ausentes ou com `gamma: 0` são tratados como no-op.
Quando todo bloco está ausente ou inativo, o circuito usa `default.qubit`
(state-vector puro). Quando qualquer slot está ativo, troca para `default.mixed`
(density matrix; ~2× mais lento, suporta canais não-unitários).
