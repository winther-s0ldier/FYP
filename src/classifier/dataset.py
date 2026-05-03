"""
Dataset loading and preprocessing for toxicity + intent classification.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional
import yaml


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# Load intent label → index mapping from config
def get_intent_labels() -> list[str]:
    cfg = load_config()
    labels = []
    for tier in ("benign", "suspicious", "malicious"):
        labels.extend(cfg["intents"][tier])
    return labels


INTENT_LABELS = get_intent_labels()
INTENT2IDX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
IDX2INTENT = {idx: label for label, idx in INTENT2IDX.items()}


class ModerationDataset(Dataset):
    """
    Unified dataset for toxicity + intent training.
    Expects a DataFrame with columns: text, toxic (0/1), intent (label string).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        has_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.has_labels:
            item["toxicity_labels"] = torch.tensor(float(row["toxic"]))
            item["intent_labels"] = torch.tensor(
                INTENT2IDX.get(row.get("intent", "question"), INTENT2IDX["question"]),
                dtype=torch.long,
            )
        return item


def load_jigsaw_2018(path: Path) -> pd.DataFrame:
    """Load Jigsaw 2018, keep text + binary toxic label."""
    df = pd.read_csv(path / "train.csv")
    df["toxic"] = (df["toxic"] >= 0.5).astype(int)
    return df[["comment_text", "toxic"]].rename(columns={"comment_text": "text"})


def load_jigsaw_2019(path: Path, sample: Optional[int] = 500_000) -> pd.DataFrame:
    """Load Jigsaw 2019 with demographic columns for fairness eval."""
    df = pd.read_csv(path / "train.csv")
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    df["toxic"] = (df["target"] >= 0.5).astype(int)

    # Keep demographic columns for fairness evaluation (Section 12.5)
    demo_cols = [c for c in df.columns if any(
        g in c for g in ["male", "female", "black", "white", "asian",
                         "hispanic", "christian", "jewish", "muslim",
                         "psychiatric", "physical_disability"]
    )]
    # Rename comment_text to text to match expected schema
    df = df.rename(columns={"comment_text": "text"})
    keep = ["text", "toxic"] + [c for c in demo_cols if c in df.columns]
    return df[keep]


def load_hateval(path: Path) -> pd.DataFrame:
    """Load HatEval 2019 (Twitter hate speech — closer to chat domain)."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        df = ds["train"].to_pandas()
        df["toxic"] = df["label"].astype(int)
        return df[["text", "toxic"]]
    except Exception:
        # Fallback: try loading raw CSV if HF dataset not available
        csv = path / "train.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["toxic"] = df.get("label", df.get("HS", 0)).astype(int)
            return df[["text", "toxic"]]
    return pd.DataFrame(columns=["text", "toxic"])


def build_train_dataset(
    tokenizer: AutoTokenizer,
    data_dir: Path = Path("data/raw"),
    max_length: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[ModerationDataset, ModerationDataset, ModerationDataset]:
    """Load, merge, split, return (train, val, test) datasets."""
    dfs = []

    jigsaw_2018 = data_dir / "jigsaw_2018"
    if (jigsaw_2018 / "train.csv").exists():
        df = load_jigsaw_2018(jigsaw_2018)
        df["source"] = "jigsaw_2018"
        dfs.append(df)
        print(f"Loaded Jigsaw 2018: {len(df):,} rows")

    jigsaw_2019 = data_dir / "jigsaw_2019"
    if (jigsaw_2019 / "train.csv").exists():
        df = load_jigsaw_2019(jigsaw_2019)
        df["source"] = "jigsaw_2019"
        dfs.append(df)
        print(f"Loaded Jigsaw 2019: {len(df):,} rows")

    hateval = data_dir / "hateval"
    if hateval.exists():
        df = load_hateval(hateval)
        df["source"] = "hateval"
        dfs.append(df)
        print(f"Loaded HatEval: {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError(
            "No datasets found in data/raw/. Run: python scripts/download_datasets.py"
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Add placeholder intent label for toxicity-only rows (to be overwritten by custom data)
    if "intent" not in combined.columns:
        combined["intent"] = combined["toxic"].map(
            lambda t: "threat" if t else "question"
        )

    # Stratified split
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(
        combined, test_size=test_ratio, stratify=combined["toxic"], random_state=seed
    )
    val_size = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["toxic"], random_state=seed
    )

    print(f"\nSplit: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    print(f"Toxic ratio — train: {train['toxic'].mean():.3f} | val: {val['toxic'].mean():.3f}")

    return (
        ModerationDataset(train, tokenizer, max_length),
        ModerationDataset(val, tokenizer, max_length),
        ModerationDataset(test, tokenizer, max_length),
    )
