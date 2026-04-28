"""
Step 1 — Download PaySim from Kaggle and save raw CSV to data/raw/.
"""
import shutil
import kagglehub
from pathlib import Path

def download():
    print("Downloading PaySim dataset from Kaggle...")
    path = kagglehub.dataset_download("ealaxi/paysim1")

    dest = Path("data/raw")
    dest.mkdir(parents=True, exist_ok=True)

    for file in Path(path).iterdir():
        shutil.copy(file, dest / file.name)
        print(f"  Copied {file.name} -> {dest / file.name}")

    csv_path = dest / "PS_20174392719_1491204439457_log.csv"
    size_mb = csv_path.stat().st_size / 1e6
    print(f"\nDone. Raw file: {csv_path} ({size_mb:.1f} MB)")
    return csv_path

if __name__ == "__main__":
    download()
