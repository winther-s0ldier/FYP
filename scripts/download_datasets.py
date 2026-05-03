"""
Download all required datasets from Kaggle.
Run from project root: python scripts/download_datasets.py

Datasets downloaded:
  - Jigsaw Toxic Comments 2018 (160K)
  - Jigsaw Unintended Bias 2019 (1.8M)
  - HatEval SemEval 2019 (13K tweets)
"""
import os
import sys
import zipfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "jigsaw-toxic-2018",
        "competition": "jigsaw-toxic-comment-classification-challenge",
        "files": ["train.csv", "test.csv", "test_labels.csv"],
        "dest": RAW_DIR / "jigsaw_2018",
    },
    {
        "name": "jigsaw-bias-2019",
        "competition": "jigsaw-unintended-bias-in-toxicity-classification",
        "files": ["train.csv", "test.csv"],
        "dest": RAW_DIR / "jigsaw_2019",
    },
]

# HatEval is part of tweet_eval
HUGGINGFACE_DATASETS = [
    {"hf_id": "tweet_eval", "config": "hate", "dest": RAW_DIR / "hateval"},
]


def setup_kaggle_auth():
    token = os.getenv("KAGGLE_TOKEN", "")
    if not token:
        print("ERROR: KAGGLE_TOKEN not set in .env")
        print("  Get your token at: https://www.kaggle.com/settings -> API")
        sys.exit(1)

    # New Kaggle auth method: set environment variables directly
    # This prevents issues with the Kaggle CLI not finding the config file
    os.environ["KAGGLE_USERNAME"] = "rudrakumar21"
    os.environ["KAGGLE_KEY"] = token
    print("Kaggle credentials injected via environment variables.")


def download_competition(competition: str, dest: Path, files: list[str]):
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in files if (dest / f).exists()]
    if len(existing) == len(files):
        print(f"  Already downloaded: {dest.name}")
        return

    print(f"  Downloading {competition}...")
    try:
        # Use the kaggle.exe directly from the venv Scripts directory
        kaggle_exe = str(Path(sys.executable).parent / "kaggle.exe")
        result = subprocess.run(
            [kaggle_exe, "competitions", "download",
             "-c", competition, "-p", str(dest)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            print(f"  Make sure you've accepted the competition rules at:")
            print(f"  https://www.kaggle.com/c/{competition}/rules")
            return

        # Unzip
        for zip_file in dest.glob("*.zip"):
            print(f"  Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(dest)
            zip_file.unlink()

        print(f"  Done: {dest}")
    except FileNotFoundError:
        print("  ERROR: 'kaggle' command not found. Run: pip install kaggle")


def download_huggingface(hf_id: str, dest: Path, config: str = None):
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"  Already downloaded: {dest.name}")
        return

    print(f"  Downloading {hf_id} (config: {config}) from HuggingFace...")
    try:
        from datasets import load_dataset
        if config:
            ds = load_dataset(hf_id, config)
        else:
            ds = load_dataset(hf_id)
        ds.save_to_disk(str(dest))
        print(f"  Done: {dest}")
    except Exception as e:
        print(f"  WARNING: HuggingFace download failed: {e}")


def verify_downloads():
    print("\nVerification:")
    all_ok = True
    checks = [
        (RAW_DIR / "jigsaw_2018" / "train.csv", "Jigsaw 2018 train"),
        (RAW_DIR / "jigsaw_2019" / "train.csv", "Jigsaw 2019 train"),
        (RAW_DIR / "hateval", "HatEval"),
    ]
    for path, name in checks:
        if path.name == "hateval":
            status = "OK" if path.exists() and any(path.iterdir()) else "X MISSING"
        else:
            status = "OK" if path.exists() else "X MISSING"
        print(f"  [{status}] {name}: {path}")
        if not path.exists():
            all_ok = False

    if not all_ok:
        print("\nSome datasets are missing. Check errors above.")
        print("You can also download manually from Kaggle and place CSVs in data/raw/")
    else:
        print("\nAll datasets ready. Next: python scripts/preprocess.py")


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)

    setup_kaggle_auth()

    for ds in DATASETS:
        print(f"\n[{ds['name']}]")
        download_competition(ds["competition"], ds["dest"], ds["files"])

    for ds in HUGGINGFACE_DATASETS:
        print(f"\n[{ds['hf_id']}]")
        download_huggingface(ds["hf_id"], ds["dest"], ds.get("config"))

    verify_downloads()
