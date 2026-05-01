"""
Step 5 — Comprehensive dataset validation before Kaggle publication.

Test categories:
  1. Schema & integrity   — nulls, types, duplicates, value ranges
  2. Balance consistency  — PIX atomicity guarantees
  3. Domain rules         — BCB regulatory features
  4. Statistical quality  — distributions, separability, KS tests
  5. Synthetic fidelity   — real vs synthetic statistical similarity
  6. ML validity          — multiple classifiers, PR-AUC, feature importance
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []

def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    tag = PASS if condition else (WARN if warn_only else FAIL)
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    results.append((tag, name))
    return condition

# ─────────────────────────────────────────────────────────
print("Loading dataset...")
df   = pd.read_parquet("data/processed/pix_fraud_br.parquet")
real = pd.read_parquet("data/processed/pix_fraud_br_real.parquet")
synth = df[~df.index.isin(real.index)].copy()  # approximate split
total = len(df)
print(f"Shape: {df.shape} | Fraud: {df['fraude'].sum():,} ({df['fraude'].mean():.4%})\n")

# ═══════════════════════════════════════════════════════════
print("-" * 60)
print("1. SCHEMA & INTEGRITY")
print("-" * 60)

EXPECTED_COLS = [
    "id_pagador", "id_recebedor", "tipo_transacao", "valor_brl",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "datetime_brasilia", "hora_dia", "dia_semana", "dia_util",
    "horario_noturno", "acima_limite_noturno",
    "razao_saldo_residual", "proporcao_valor_recebedor", "fraude",
]

check("All expected columns present",
      set(EXPECTED_COLS) == set(df.columns),
      f"Missing: {set(EXPECTED_COLS) - set(df.columns)}")

check("No null values", df.isnull().sum().sum() == 0,
      f"Nulls: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")

check("No duplicate rows", df.duplicated().sum() == 0,
      f"Duplicates: {df.duplicated().sum():,}")

check("Exactly 2,000,000 rows", total == 2_000_000, f"Got: {total:,}")

check("fraude is binary (0/1)",
      set(df["fraude"].unique()) == {0, 1})

check("hora_dia in [0, 23]",
      df["hora_dia"].between(0, 23).all(),
      f"Out of range: {(~df['hora_dia'].between(0, 23)).sum()}")

check("valor_brl > 0 for all rows",
      (df["valor_brl"] > 0).all(),
      f"Zero/negative: {(df['valor_brl'] <= 0).sum()}")

valid_tipos = {"chave_pix", "dados_bancarios", "pix_copia_e_cola"}
check("tipo_transacao only valid BCB modalities",
      set(df["tipo_transacao"].unique()) == valid_tipos,
      f"Found: {set(df['tipo_transacao'].unique())}")

# Check 7 unique values covering all days — encoding-safe (avoids cp1252 issues)
check("dia_semana has exactly 7 unique values (all weekdays covered)",
      df["dia_semana"].nunique() == 7,
      f"Found {df['dia_semana'].nunique()} unique values: {sorted(df['dia_semana'].unique())}")

check("datetime_brasilia within Jan 2024",
      (df["datetime_brasilia"] >= "2024-01-01").all() and
      (df["datetime_brasilia"] <= "2024-01-31 23:59:59").all())

# ═══════════════════════════════════════════════════════════
print()
print("-" * 60)
print("2. BALANCE CONSISTENCY (PIX ATOMICITY)")
print("-" * 60)

check("No negative posterior pagador",
      (df["saldo_posterior_pagador"] >= 0).all(),
      f"Negative: {(df['saldo_posterior_pagador'] < 0).sum()}")

check("No negative posterior recebedor",
      (df["saldo_posterior_recebedor"] >= 0).all(),
      f"Negative: {(df['saldo_posterior_recebedor'] < 0).sum()}")

check("posterior_recebedor >= anterior_recebedor",
      (df["saldo_posterior_recebedor"] >= df["saldo_anterior_recebedor"]).all(),
      f"Violations: {(df['saldo_posterior_recebedor'] < df['saldo_anterior_recebedor']).sum()}")

delta_err = (df["saldo_anterior_pagador"] - df["saldo_posterior_pagador"] - df["valor_brl"]).abs()
check("posterior_pagador = anterior_pagador - valor (tol R$0.01)",
      (delta_err <= 0.01).all(),
      f"Violations: {(delta_err > 0.01).sum()}")

check("razao_saldo_residual in [0, 1]",
      df["razao_saldo_residual"].between(0, 1).all(),
      f"Out of range: {(~df['razao_saldo_residual'].between(0, 1)).sum()}")

check("proporcao_valor_recebedor in [0, 1]",
      df["proporcao_valor_recebedor"].between(0, 1).all(),
      f"Out of range: {(~df['proporcao_valor_recebedor'].between(0, 1)).sum()}")

# ═══════════════════════════════════════════════════════════
print()
print("-" * 60)
print("3. DOMAIN RULES (BCB)")
print("-" * 60)

noturno_computed = (df["hora_dia"] >= 20) | (df["hora_dia"] < 6)
check("horario_noturno consistent with hora_dia",
      (df["horario_noturno"] == noturno_computed).all(),
      f"Inconsistent: {(df['horario_noturno'] != noturno_computed).sum()}")

acima_computed = (df["valor_brl"] > 1_000) & df["horario_noturno"]
check("acima_limite_noturno consistent with valor_brl + horario_noturno",
      (df["acima_limite_noturno"] == acima_computed).all(),
      f"Inconsistent: {(df['acima_limite_noturno'] != acima_computed).sum()}")

noturno_pct = df["horario_noturno"].mean()
check("horario_noturno ~20-25% of transactions (10h out of 24h)",
      0.15 <= noturno_pct <= 0.30,
      f"Got: {noturno_pct:.1%}", warn_only=True)

fraud_rate = df["fraude"].mean()
check("Fraud rate between 0.5% and 1.5% (realistic range)",
      0.005 <= fraud_rate <= 0.015,
      f"Got: {fraud_rate:.4%}")

check("Fraud only in TRANSFER modalities (no PAYMENT types)",
      not any(t in df["tipo_transacao"].unique()
              for t in ["qr_code_dinamico", "qr_code_estatico", "pix_aproximacao"]))

# ═══════════════════════════════════════════════════════════
print()
print("-" * 60)
print("4. STATISTICAL QUALITY")
print("-" * 60)

# Separability per key feature (KS statistic)
for feat, threshold in [
    ("razao_saldo_residual", 0.40),
    ("saldo_anterior_recebedor", 0.50),
    ("saldo_anterior_pagador", 0.15),
]:
    legit = df[df["fraude"] == 0][feat]
    fraud = df[df["fraude"] == 1][feat]
    ks, pval = stats.ks_2samp(legit.sample(5000, random_state=42),
                               fraud.sample(min(5000, len(fraud)), random_state=42))
    check(f"KS({feat}) >= {threshold} (strong class separation)",
          ks >= threshold,
          f"KS={ks:.4f}, p={pval:.2e}")

# Tipo transacao distribution
tipo_dist = df["tipo_transacao"].value_counts(normalize=True)
check("chave_pix is most frequent modality (>50%)",
      tipo_dist.get("chave_pix", 0) > 0.50,
      f"chave_pix share: {tipo_dist.get('chave_pix', 0):.1%}")

# Fraud distribution across modalities — all should have fraud
fraud_by_tipo = df.groupby("tipo_transacao")["fraude"].mean()
check("All modalities have fraud cases",
      (fraud_by_tipo > 0).all(),
      f"Fraud rates: {fraud_by_tipo.round(4).to_dict()}")

# ═══════════════════════════════════════════════════════════
print()
print("-" * 60)
print("5. SYNTHETIC FIDELITY (real vs synthetic)")
print("-" * 60)

# Use real rows vs synthetic rows from the 2M dataset
# Real: first 532,909 in pix_fraud_br_real.parquet
real_sample = real.sample(min(10_000, len(real)), random_state=42)
synth_sample = df[~df[["id_pagador","id_recebedor","datetime_brasilia"]].apply(tuple, axis=1).isin(
    real[["id_pagador","id_recebedor","datetime_brasilia"]].apply(tuple, axis=1)
)].sample(10_000, random_state=42)

for feat in ["valor_brl", "saldo_anterior_pagador", "saldo_anterior_recebedor", "razao_saldo_residual"]:
    ks, pval = stats.ks_2samp(real_sample[feat], synth_sample[feat])
    # KS should be LOW (distributions similar) — warn if > 0.10
    check(f"KS({feat}) real vs synthetic < 0.10 (distributions similar)",
          ks < 0.10,
          f"KS={ks:.4f} — {'distributions diverge' if ks >= 0.10 else 'OK'}",
          warn_only=ks >= 0.10)

real_fraud_rate  = real_sample["fraude"].mean()
synth_fraud_rate = synth_sample["fraude"].mean()
check("Synthetic fraud rate matches real (±0.3%)",
      abs(real_fraud_rate - synth_fraud_rate) < 0.003,
      f"Real: {real_fraud_rate:.4%} | Synthetic: {synth_fraud_rate:.4%}",
      warn_only=True)

# ═══════════════════════════════════════════════════════════
print()
print("-" * 60)
print("6. ML VALIDITY (multiple classifiers, 100k sample)")
print("-" * 60)

le_tipo = LabelEncoder()
le_dia  = LabelEncoder()
df["tipo_enc"] = le_tipo.fit_transform(df["tipo_transacao"])
df["dia_enc"]  = le_dia.fit_transform(df["dia_semana"])

FEATURES = [
    "valor_brl",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "hora_dia", "dia_util", "horario_noturno", "acima_limite_noturno",
    "razao_saldo_residual", "proporcao_valor_recebedor",
    "tipo_enc", "dia_enc",
]

X_s, _, y_s, _ = train_test_split(
    df[FEATURES].astype(float), df["fraude"],
    train_size=100_000, stratify=df["fraude"], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
)
sw = (y_tr == 0).sum() / (y_tr == 1).sum()

classifiers = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, scale_pos_weight=sw,
        random_state=42, eval_metric="aucpr", verbosity=0),
}

for name, clf in classifiers.items():
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, prob)
    prauc = average_precision_score(y_te, prob)
    check(f"{name}: ROC-AUC >= 0.90",  auc >= 0.90,   f"ROC-AUC={auc:.4f} | PR-AUC={prauc:.4f}")
    check(f"{name}: PR-AUC  >= 0.50",  prauc >= 0.50, f"ROC-AUC={auc:.4f} | PR-AUC={prauc:.4f}")

# Feature importance stability (XGBoost)
xgb = classifiers["XGBoost"]
imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
top2 = imp.index[:2].tolist()
check("razao_saldo_residual in top 3 features",
      "razao_saldo_residual" in imp.index[:3].tolist(),
      f"Top 3: {imp.index[:3].tolist()}")
check("saldo_anterior_recebedor in top 3 features",
      "saldo_anterior_recebedor" in imp.index[:3].tolist(),
      f"Top 3: {imp.index[:3].tolist()}")

# ═══════════════════════════════════════════════════════════
print()
print("-" * 60)
print("SUMMARY")
print("-" * 60)
passed = sum(1 for t, _ in results if t == PASS)
warned = sum(1 for t, _ in results if t == WARN)
failed = sum(1 for t, _ in results if t == FAIL)
total_checks = len(results)

print(f"  Total checks : {total_checks}")
print(f"  {PASS}        : {passed}")
print(f"  {WARN}        : {warned}")
print(f"  {FAIL}        : {failed}")
print()
if failed == 0:
    print("  Dataset is ready for publication.")
else:
    print("  Fix FAIL items before publishing.")
    for tag, name in results:
        if tag == FAIL:
            print(f"    - {name}")
