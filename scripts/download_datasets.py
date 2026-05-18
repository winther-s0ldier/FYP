"""
Download all required datasets for Phase 1 + Phase 2 training.
Run from project root: python scripts/download_datasets.py

Phase 1 (Kaggle):
  - Jigsaw Toxic Comments 2018 (160K)
  - Jigsaw Unintended Bias 2019 (1.8M)

Phase 2 (HuggingFace):
  - HatEval SemEval 2019 (13K tweets)
  - ToxiGen (274K implicit hate)             — already downloaded
  - Civil Comments (1.8M)                    — already downloaded
  - MetaHate (1.7M)                          — already downloaded
  - Banking77 (13K intent classification)    — NEW
  - CLINC150 (23K intent + OOS detection)    — NEW
  - DailyDialog (13K dialog acts)            — NEW
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

# HuggingFace datasets (Phase 1 + Phase 2)
HUGGINGFACE_DATASETS = [
    # Phase 1
    {"hf_id": "tweet_eval", "config": "hate", "dest": RAW_DIR / "hateval"},
    # Phase 2 — Intent Pre-training (fixes Question Bias)
    {"hf_id": "legacy-datasets/banking77", "config": None, "dest": RAW_DIR / "banking77"},
    {"hf_id": "clinc_oos", "config": "plus", "dest": RAW_DIR / "clinc150"},
    {"hf_id": "DeepPavlov/daily_dialog", "config": None, "dest": RAW_DIR / "daily_dialog"},
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
        # Detect the correct kaggle executable (Linux vs Windows)
        bin_dir = Path(sys.executable).parent
        kaggle_exe = bin_dir / "kaggle"
        if not kaggle_exe.exists():
            kaggle_exe = bin_dir / "kaggle.exe"
            
        if not kaggle_exe.exists():
            # Fallback to system path if not in venv
            kaggle_exe = "kaggle"

        result = subprocess.run(
            [str(kaggle_exe), "competitions", "download",
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
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    all_ok = True

    # File-based checks (CSVs)
    file_checks = [
        (RAW_DIR / "jigsaw_2018" / "train.csv", "Jigsaw 2018"),
        (RAW_DIR / "jigsaw_2019" / "train.csv", "Jigsaw 2019"),
        (RAW_DIR / "metahate" / "available_metahate.tsv", "MetaHate"),
    ]
    for path, name in file_checks:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not path.exists():
            all_ok = False

    # Directory-based checks (HF datasets / parquet)
    dir_checks = [
        (RAW_DIR / "hateval", "HatEval"),
        (RAW_DIR / "toxigen", "ToxiGen"),
        (RAW_DIR / "civil_comments", "Civil Comments"),
        (RAW_DIR / "banking77", "Banking77"),
        (RAW_DIR / "clinc150", "CLINC150"),
        (RAW_DIR / "daily_dialog", "DailyDialog"),
    ]
    for path, name in dir_checks:
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\nSome datasets are missing. Check errors above.")
        print("You can also download manually and place files in data/raw/")
    else:
        print("\nAll datasets ready!")
        print("  Stage 1: python -m src.classifier.train --stage 1")
        print("  Stage 2: python -m src.classifier.train --stage 2")


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
