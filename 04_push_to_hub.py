"""
Step 4 — Push processed dataset to Hugging Face Hub.

Usage:
    python 04_push_to_hub.py --token hf_xxxxxxxxxxxxxxxxxxxx

Get your token at: https://huggingface.co/settings/tokens (role: Write)
"""
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset
from huggingface_hub import login

PARQUET_FILE = Path("data/processed/pix_fraud_br.parquet")
HF_REPO_ID   = "andremessina/pix-fraud-br"


def push(token: str):
    print("Authenticating with Hugging Face...")
    login(token=token)
    print("  Authenticated.")

    print(f"\nLoading {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"  Shape: {df.shape} | Fraud rate: {df['fraude'].mean():.4%}")

    print("\nConverting to HuggingFace Dataset...")
    hf_dataset = Dataset.from_pandas(df, preserve_index=False)

    print(f"Pushing to {HF_REPO_ID}...")
    hf_dataset.push_to_hub(
        HF_REPO_ID,
        private=False,
        commit_message="Add PIX-BR fraud detection dataset v1.0",
    )

    print(f"\nDataset published: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Hugging Face write token")
    args = parser.parse_args()
    push(args.token)
