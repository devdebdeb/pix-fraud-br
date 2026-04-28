"""
Step 3 — Synthetic data generation via stratified bootstrap + Gaussian noise.

Consistency rules enforced:
    saldo_posterior_pagador   = saldo_anterior_pagador   - valor_brl  (>= 0)
    saldo_posterior_recebedor = saldo_anterior_recebedor + valor_brl

Only saldo_anterior_* and valor_brl receive noise. Posteriors are always derived,
never noised independently — this is what caused the 12% inconsistency bug before.
"""
import numpy as np
import pandas as pd
from pathlib import Path

REAL_FILE  = Path("data/processed/pix_fraud_br_real.parquet")
OUT_FILE   = Path("data/processed/pix_fraud_br.parquet")
TARGET     = 2_000_000
SEED       = 42
NOISE_FRAC = 0.05

# Noised independently: valor, saldo anteriores, hora
BASE_NUM_COLS = ["valor_brl", "saldo_anterior_pagador", "saldo_anterior_recebedor"]


def synthesize_class(df_class: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx   = rng.integers(0, len(df_class), size=n)
    synth = df_class.iloc[idx].copy().reset_index(drop=True)

    # Noise on base numerical columns
    stds = df_class[BASE_NUM_COLS].std()
    for col in BASE_NUM_COLS:
        noise = rng.normal(0, stds[col] * NOISE_FRAC, size=n)
        synth[col] = (synth[col].values + noise).clip(min=0.01).round(2)

    # Ensure saldo_anterior_pagador > valor_brl (k >= 1.001)
    min_saldo = synth["valor_brl"] * 1.001
    synth["saldo_anterior_pagador"] = synth["saldo_anterior_pagador"].clip(lower=min_saldo)

    # Derive posteriors — guaranteed consistent
    synth["saldo_posterior_pagador"]   = (synth["saldo_anterior_pagador"] - synth["valor_brl"]).round(2)
    synth["saldo_posterior_recebedor"] = (synth["saldo_anterior_recebedor"] + synth["valor_brl"]).round(2)

    # hora_dia: integer noise ±2h
    synth["hora_dia"] = ((synth["hora_dia"].values + rng.integers(-2, 3, size=n)) % 24).astype(int)

    # Re-derive features that depend on hora_dia or noised values
    synth["horario_noturno"]    = (synth["hora_dia"] >= 20) | (synth["hora_dia"] < 6)
    synth["dia_util"]           = synth["dia_semana"].isin([
        "segunda-feira", "terca-feira", "quarta-feira", "quinta-feira", "sexta-feira"
    ])
    synth["acima_limite_noturno"]       = (synth["valor_brl"] > 1_000) & synth["horario_noturno"]
    synth["razao_saldo_residual"]       = (synth["saldo_posterior_pagador"] / synth["saldo_anterior_pagador"]).round(6)
    synth["proporcao_valor_recebedor"]  = (
        synth["valor_brl"] / (synth["valor_brl"] + synth["saldo_anterior_recebedor"])
    ).round(6)

    # Shift datetime ±3 days to avoid exact duplicates
    day_offsets = pd.to_timedelta(rng.integers(-3, 4, size=n), unit="D")
    dt = (pd.to_datetime(synth["datetime_brasilia"]) + day_offsets).clip(
        lower=pd.Timestamp("2024-01-01"),
        upper=pd.Timestamp("2024-01-31 23:59:59"),
    )
    synth["datetime_brasilia"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")

    # New synthetic IDs
    a = rng.integers(100, 999, n).astype(str)
    b = rng.integers(100, 999, n).astype(str)
    synth["id_pagador"]   = ["***." + x + "." + y + "-**" for x, y in zip(a, b)]
    c = rng.integers(100, 999, n).astype(str)
    d = rng.integers(100, 999, n).astype(str)
    synth["id_recebedor"] = ["***." + x + "." + y + "-**" for x, y in zip(c, d)]

    return synth


def synthesize():
    rng = np.random.default_rng(SEED)

    print(f"Loading {REAL_FILE}...")
    real = pd.read_parquet(REAL_FILE)
    n_real      = len(real)
    n_synth     = TARGET - n_real
    fraud_rate  = real["fraude"].mean()

    n_synth_fraud = round(n_synth * fraud_rate)
    n_synth_legit = n_synth - n_synth_fraud

    print(f"  Real rows:      {n_real:,} (fraud rate: {fraud_rate:.4%})")
    print(f"  Synthetic rows: {n_synth:,} (fraud: {n_synth_fraud:,} | legit: {n_synth_legit:,})")

    df_fraud = real[real["fraude"] == 1]
    df_legit = real[real["fraude"] == 0]

    print("\nGenerating synthetic fraud rows...")
    synth_fraud         = synthesize_class(df_fraud, n_synth_fraud, rng)
    synth_fraud["fraude"] = 1

    print("Generating synthetic legitimate rows...")
    synth_legit         = synthesize_class(df_legit, n_synth_legit, rng)
    synth_legit["fraude"] = 0

    print("Combining and shuffling...")
    combined = pd.concat([real, synth_fraud, synth_legit], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    combined["fraude"]               = combined["fraude"].astype(int)
    combined["hora_dia"]             = combined["hora_dia"].astype(int)
    combined["dia_util"]             = combined["dia_util"].astype(bool)
    combined["horario_noturno"]      = combined["horario_noturno"].astype(bool)
    combined["acima_limite_noturno"] = combined["acima_limite_noturno"].astype(bool)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_FILE, index=False)

    print(f"\nFinal dataset: {combined.shape}")
    print(f"Fraud rate:    {combined['fraude'].mean():.4%}")
    print(f"Saved:         {OUT_FILE}")


if __name__ == "__main__":
    synthesize()
