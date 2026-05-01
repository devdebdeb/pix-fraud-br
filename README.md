---
license: odc-by
task_categories:
  - tabular-classification
tags:
  - fraud-detection
  - finance
  - pix
  - brazil
  - synthetic
  - portuguese
  - imbalanced-learning
pretty_name: PIX Fraud BR
size_categories:
  - 1M<n<10M
language:
  - pt
---

# PIX Fraud BR

A synthetic dataset of **2 million PIX instant payment transactions** in the Brazilian financial system, designed for fraud detection research. Derived from the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile money simulator and adapted to reflect Brazil's official PIX modalities, BCB regulatory rules, and real-world fraud patterns documented by Febraban.

| | |
|---|---|
| **Rows** | 2,000,000 |
| **Fraud cases** | 15,376 (0.77%) |
| **Features** | 17 |
| **Real rows** (PaySim TRANSFER, all 532,909 kept) | 532,909 |
| **Synthetic rows** | 1,467,091 |
| **PaySim source** | 6,362,620 total rows — only TRANSFER (532,909) used |
| **Period** | January 2024 (BRT, UTC-3) |

---

## Background

**PIX** is Brazil's instant payment system, operated by the Banco Central do Brasil (BCB). Launched in November 2020, it processed over **63.8 billion transactions in 2024** — surpassing the combined volume of credit cards, debit cards, boleto, and TED. The rapid adoption has made it the primary target for social engineering fraud in Brazil, where fraudsters manipulate victims into authorizing transfers to mule accounts (*contas laranja*).

No public dataset of real PIX fraud transactions exists — financial institutions treat this data as proprietary. This dataset fills that gap by adapting PaySim's transaction simulator to match the PIX ecosystem.

---

## Design Decisions

### Transaction types

PaySim simulates five transaction types. Only `TRANSFER` was kept:

| PaySim type | Decision | Reason |
|---|---|---|
| `TRANSFER` | **Kept** | P2P transfers between individuals — the only type with fraud in PaySim, and the primary PIX fraud vector |
| `PAYMENT` | Dropped | Recipients are merchant accounts with untracked balances (0/0 in 100% of cases) — pure noise |
| `CASH-IN`, `CASH-OUT`, `DEBIT` | Dropped | No direct PIX equivalent |

### PIX modalities

`tipo_transacao` follows BCB's [Manual de Padrões para Iniciação do Pix v2.9.0](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/II_ManualdePadroesparaIniciacaodoPix.pdf), assigned with realistic P2P distributions:

| Value | Description | Share |
|---|---|---|
| `chave_pix` | Transfer by PIX key (CPF, phone, email, random) | 68% |
| `dados_bancarios` | Transfer by manual bank data (branch + account) | 20% |
| `pix_copia_e_cola` | Copy-and-paste of QR Code payload | 12% |

### Balance generation

PaySim's original balance columns track **cumulative account state per time step**, not the isolated before/after state of a single transaction — making them inconsistent in 98% of rows. All balance values were regenerated synthetically from `valor_brl` using distributions conditioned on fraud label:

**Sender (pagador):**
- Legitimate: `k ~ LogNormal(1.5, 0.6)` → median k ≈ 4.5x → pays ~22% of balance
- Fraud: `k ~ LogNormal(0.05, 0.3)` → median k ≈ 1.05x → empties ~95% of balance

**Receiver (recebedor):**
- Legitimate: `LogNormal(10.0, 1.0)` → median ≈ R$22k (normal account balance)
- Fraud (mule account): `LogNormal(6.0, 1.5)` → median ≈ R$400 (low pre-existing balance)

Consistency is enforced:
```
saldo_posterior_pagador   = saldo_anterior_pagador   − valor_brl
saldo_posterior_recebedor = saldo_anterior_recebedor + valor_brl
```

---

## Schema

| Column | Type | Description |
|---|---|---|
| `id_pagador` | string | Masked CPF of sender (`***.XXX.XXX-**`) |
| `id_recebedor` | string | Masked CPF of receiver (`***.XXX.XXX-**`) |
| `tipo_transacao` | string | PIX initiation modality (see above) |
| `valor_brl` | float | Transaction amount in BRL |
| `saldo_anterior_pagador` | float | Sender balance before transaction |
| `saldo_posterior_pagador` | float | Sender balance after transaction |
| `saldo_anterior_recebedor` | float | Receiver balance before transaction |
| `saldo_posterior_recebedor` | float | Receiver balance after transaction |
| `datetime_brasilia` | string | Transaction datetime in BRT (UTC-3) |
| `hora_dia` | int | Hour of day (0–23) |
| `dia_semana` | string | Day of week in Portuguese |
| `dia_util` | bool | True if Monday–Friday |
| `horario_noturno` | bool | True if 20h–06h (BCB default night period) |
| `acima_limite_noturno` | bool | True if `valor_brl > R$1,000` and `horario_noturno` |
| `razao_saldo_residual` | float | `saldo_posterior_pagador / saldo_anterior_pagador` — proportion of balance remaining |
| `proporcao_valor_recebedor` | float | `valor_brl / (valor_brl + saldo_anterior_recebedor)` — transaction weight in receiver's total |
| **`fraude`** | int | **Target — 1 = fraud, 0 = legitimate** |

---

## Fraud Patterns

Fraud reflects the most common PIX attack in Brazil: social engineering or account takeover, followed by transfer to a *conta laranja* (mule account).

| Signal | Legitimate | Fraud |
|---|---|---|
| `razao_saldo_residual` median | **0.78** (paid 22%) | **0.08** (emptied 92%) |
| `saldo_anterior_recebedor` median | **R$22,152** | **R$440** |
| `saldo_anterior_recebedor` IQR | R$11k – R$43k | R$151 – R$1,107 |

### Multi-model validation (100k sample, class-balanced)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.9926 | 0.5420 |
| Random Forest | 0.9816 | 0.8221 |
| Gradient Boosting | 0.9863 | 0.7622 |
| XGBoost | 0.9933 | 0.8053 |

All four models exceed ROC-AUC 0.98 and PR-AUC 0.54. The gap between Logistic Regression and tree-based models in PR-AUC reflects the non-linear interaction between `saldo_anterior_recebedor` and `razao_saldo_residual`.

**Feature importances (XGBoost, top):**

| Feature | Importance | Interpretation |
|---|---|---|
| `saldo_anterior_recebedor` | 68.1% | Mule accounts have ~50× lower balance than legitimate receivers |
| `razao_saldo_residual` | 10.5% | Fraudulent senders nearly empty their accounts |
| `saldo_anterior_pagador` | 5.7% | Supporting signal for sender profile |
| `proporcao_valor_recebedor` | 3.1% | Transaction dominates mule account's total funds |

---

## Class Imbalance

The 0.77% fraud rate is intentional — realistic fraud rates in payment systems are low by design:

| Dataset | Fraud rate |
|---|---|
| Real payment systems (BCB estimates) | 0.01–0.1% |
| Kaggle Credit Card Fraud (benchmark) | 0.17% |
| **PIX Fraud BR** | **0.77%** |

Users should apply imbalanced learning techniques: `class_weight='balanced'`, SMOTE, threshold tuning, and evaluate with **PR-AUC** rather than accuracy.

---

## Validation

A comprehensive validation suite (`05_validate.py`) covers 41 checks across 6 categories: schema & integrity, balance consistency, BCB domain rules, statistical quality, synthetic fidelity, and ML validity. Results: **40 PASS, 0 FAIL, 1 WARN** (KS for `razao_saldo_residual` real vs synthetic = 0.104, marginally above threshold of 0.10 — base columns all pass with KS < 0.05).

---

## Synthetic Data Methodology

All 532,909 TRANSFER rows from PaySim (out of 6,362,620 total) were kept as the real base. 1,467,091 synthetic rows were generated via **stratified bootstrap with Gaussian noise**:

1. Rows are sampled with replacement, stratified by `fraude`
2. Gaussian noise (5% of each column's std) is added to `valor_brl`, `saldo_anterior_pagador`, `saldo_anterior_recebedor`
3. Posterior balances are re-derived to enforce consistency
4. Temporal features, risk ratios, and IDs are re-derived from noised base values
5. `datetime_brasilia` is shifted ±3 days to avoid exact duplicates

---

## Known Limitations

- **Transaction amounts**: `valor_brl` inherits PaySim's distribution (median R$12k, mean R$191k), which is larger than typical Brazilian retail PIX transactions. This limits the discriminative power of `proporcao_valor_recebedor`.
- **Temporal scope**: Simulates a single month (January 2024). Does not capture seasonal patterns.
- **Account IDs**: Fully anonymized — cannot be used for graph-based (network) fraud detection.
- **No multi-hop transfers**: Real *conta laranja* networks often involve chains of 2–4 accounts. This dataset models only the first transfer.

---

## Source & License

Derived from [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) by Edgar Lopez-Rojas et al., licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Regulatory references:**
- [BCB — Manual de Padrões para Iniciação do Pix v2.9.0](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/II_ManualdePadroesparaIniciacaodoPix.pdf)
- [BCB — Resolução BCB nº 6/2020 — Limites noturno](https://www.bcb.gov.br/)
- [Febraban — Golpes e fraudes com PIX (2024)](https://portal.febraban.org.br/noticia/3903/pt-br/)
- [Febraban — Autorregulação contas laranja (2024)](https://portal.febraban.org.br/noticia/4367/pt-br/)

---

## Citation

```bibtex
@dataset{messina2025pixfraudbr,
  author    = {Messina, André},
  title     = {PIX Fraud BR: Synthetic Brazilian PIX Fraud Detection Dataset},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/andremessina/pix-fraud-br},
  note      = {Derived from PaySim (Lopez-Rojas et al., 2016). Balance columns regenerated synthetically.}
}
```
