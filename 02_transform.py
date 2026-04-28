"""
Step 2 — Transform PaySim into PIX-BR fraud dataset (P2P only).

Decisions documented:
- Keep only TRANSFER (P2P): única modalidade com fraude no PaySim e
  equivalente direto ao PIX entre pessoas físicas
- Drop PAYMENT/CASH-IN/CASH-OUT/DEBIT: sem equivalente PIX ou sem sinal de fraude
- Drop isFlaggedFraud: regra hardcoded PaySim >200k, sem equivalente BCB
- Drop saldos originais do PaySim: PaySim rastreia saldo agregado de conta por step,
  não o estado isolado de uma transação — 98% das linhas inconsistentes
- Regenerar saldos sinteticamente a partir de valor_brl com distribuições
  condicionadas em fraude/legítima (ver regenerate_balances)
- Tipo transacao: modalidades BCB do Manual de Padrões v2.9.0
- Temporal: step (1h) → BRT UTC-3, base 2024-01-01
- Risco: features derivadas das regras BCB e padrões Febraban
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_FILE  = Path("data/raw/PS_20174392719_1491204439457_log.csv")
OUT_FILE  = Path("data/processed/pix_fraud_br_real.parquet")
SEED      = 42
BASE_DATE = pd.Timestamp("2024-01-01", tz="America/Sao_Paulo")

TRANSFER_MODALITIES = ["chave_pix", "dados_bancarios", "pix_copia_e_cola"]
TRANSFER_WEIGHTS    = [0.68, 0.20, 0.12]

# Parâmetros para geração de saldos (LogNormal: k = saldo_anterior / valor_brl)
# Legítima: k ~ exp(N(1.5, 0.6)) → mediana ≈ 4.5x → paga ~22% do saldo
# Fraude:   k ~ exp(N(0.05, 0.3)) → mediana ≈ 1.05x → esvazia ~95% do saldo
K_PAG_LEGIT  = (1.50, 0.60)
K_PAG_FRAUD  = (0.05, 0.30)

# Saldo recebedor (LogNormal direto em BRL)
# Legítima: conta normal → mediana ≈ R$22k (exp(10)), σ=1.0 concentra a distribuição
#           evita caudas muito baixas que tornam proporcao_valor_recebedor ineficaz
#           p5 ≈ R$4.2k (exp(8.35)) vs p5 ≈ R$820 com σ=2.0
# Fraude:   conta laranja → mediana ≈ R$400 (exp(6)), mais dispersa (σ=1.5)
REC_LEGIT    = (10.0, 1.0)
REC_FRAUD    = ( 6.0, 1.5)


def load_and_filter(path: Path) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")
    df = df[df["type"] == "TRANSFER"].copy()
    print(f"  After filter (TRANSFER only): {df.shape}")
    print(f"  Fraud rate: {df['isFraud'].mean():.4%}")
    return df


def add_tipo_transacao(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df["tipo_transacao"] = rng.choice(
        TRANSFER_MODALITIES, size=len(df), p=TRANSFER_WEIGHTS
    )
    return df


def add_account_ids(df: pd.DataFrame) -> pd.DataFrame:
    def to_cpf(uid: str) -> str:
        digits = uid[-9:].zfill(9)
        return f"***.{digits[:3]}.{digits[3:6]}-**"

    df["id_pagador"]  = df["nameOrig"].map(to_cpf)
    df["id_recebedor"] = df["nameDest"].map(to_cpf)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = BASE_DATE + pd.to_timedelta(df["step"], unit="h")
    df["datetime_brasilia"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["hora_dia"]          = dt.dt.hour
    df["dia_semana"]        = dt.dt.day_name(locale="pt_BR.UTF-8").str.lower()
    df["dia_util"]          = dt.dt.dayofweek < 5
    df["horario_noturno"]   = (df["hora_dia"] >= 20) | (df["hora_dia"] < 6)
    return df


def regenerate_balances(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Gera saldos sinteticamente a partir de valor_brl com consistência garantida:
        saldo_posterior_pagador   = saldo_anterior_pagador   - valor_brl  (>= 0)
        saldo_posterior_recebedor = saldo_anterior_recebedor + valor_brl

    Distribuições condicionadas em isFraud (padrões Febraban):
    - Legítima: saldo_anterior_pagador >> valor_brl (pagador tem folga)
    - Fraude:   saldo_anterior_pagador ≈ valor_brl  (conta esvaziada)
    - Fraude:   saldo_anterior_recebedor baixo      (conta laranja)
    """
    valor      = df["amount"].values
    fraud_mask = df["isFraud"].values == 1

    # --- Pagador: k = saldo_anterior / valor_brl, forçado >= 1.001 ---
    mu_k  = np.where(fraud_mask, K_PAG_FRAUD[0], K_PAG_LEGIT[0])
    sig_k = np.where(fraud_mask, K_PAG_FRAUD[1], K_PAG_LEGIT[1])
    k     = np.exp(rng.normal(mu_k, sig_k)).clip(min=1.001)

    saldo_ant_pag = (valor * k).round(2)
    saldo_pos_pag = (saldo_ant_pag - valor).round(2)

    # --- Recebedor: saldo pré-existente, independente do valor ---
    mu_r  = np.where(fraud_mask, REC_FRAUD[0], REC_LEGIT[0])
    sig_r = np.where(fraud_mask, REC_FRAUD[1], REC_LEGIT[1])

    saldo_ant_rec = np.exp(rng.normal(mu_r, sig_r)).round(2)
    saldo_pos_rec = (saldo_ant_rec + valor).round(2)

    df["saldo_anterior_pagador"]   = saldo_ant_pag
    df["saldo_posterior_pagador"]  = saldo_pos_pag
    df["saldo_anterior_recebedor"] = saldo_ant_rec
    df["saldo_posterior_recebedor"]= saldo_pos_rec
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    # BCB: sinal de risco para transacoes > R$1.000 no periodo noturno
    df["acima_limite_noturno"] = (df["amount"] > 1_000) & df["horario_noturno"]

    # Proporção do saldo restante após a transação — [0, 1]
    # Fraude: ~0.05 (sobrou 5% — conta quase esvaziada)
    # Legítima: ~0.78 (sobrou 78% — transação parcial normal)
    df["razao_saldo_residual"] = (
        df["saldo_posterior_pagador"] / df["saldo_anterior_pagador"]
    ).round(6)

    # Proporção do valor recebido no total (valor + saldo pré) do recebedor — [0, 1]
    # Conta laranja: saldo baixo → valor domina → ratio → 1.0
    # Legítima: saldo alto (σ=1.0 concentrado) → saldo domina → ratio → 0.0
    # Formulação limitada [0,1]: sem divisão por zero, sem valores extremos
    df["proporcao_valor_recebedor"] = (
        df["amount"] / (df["amount"] + df["saldo_anterior_recebedor"])
    ).round(6)

    return df


def rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "amount":  "valor_brl",
        "isFraud": "fraude",
    })

    return df[[
        "id_pagador", "id_recebedor",
        "tipo_transacao", "valor_brl",
        "saldo_anterior_pagador", "saldo_posterior_pagador",
        "saldo_anterior_recebedor", "saldo_posterior_recebedor",
        "datetime_brasilia", "hora_dia", "dia_semana", "dia_util", "horario_noturno",
        "acima_limite_noturno", "razao_saldo_residual", "proporcao_valor_recebedor",
        "fraude",
    ]]


def transform():
    rng = np.random.default_rng(SEED)

    df = load_and_filter(RAW_FILE)
    df = add_tipo_transacao(df, rng)
    df = add_account_ids(df)
    df = add_temporal_features(df)
    df = regenerate_balances(df, rng)
    df = add_risk_features(df)
    df = rename_and_select(df)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    print(f"\nDataset real: {df.shape}")
    print(f"Fraud rate:   {df['fraude'].mean():.4%}")
    print(f"Saved:        {OUT_FILE}")
    return df


if __name__ == "__main__":
    transform()
