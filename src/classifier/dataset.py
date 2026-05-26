"""
Dataset loading and preprocessing for toxicity + intent classification.
Phase 2: Supports all 7 training sources + harmonization to 20-label taxonomy.
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


# ---------------------------------------------------------------------------
# Text normalisation — emoji → text description
# ---------------------------------------------------------------------------
try:
    import emoji as _emoji_lib

    def normalize_text(text: str) -> str:
        """
        Convert emojis to their text descriptions so BPE tokenizes them as
        meaningful subwords rather than fragmented byte sequences.

        Examples:
          "i will get you 🔪"      → "i will get you  kitchen_knife "
          "you ok? 🙂"             → "you ok?  slightly_smiling_face "
          "grooming u ❤️ 😘"      → "grooming u  red_heart   face_blowing_a_kiss "

        Called at dataset load time so training and inference are consistent.
        """
        return _emoji_lib.demojize(str(text), delimiters=(" ", " "))

except ImportError:
    def normalize_text(text: str) -> str:  # type: ignore[misc]
        """Fallback: no-op if emoji library is not installed."""
        return str(text)


# ---------------------------------------------------------------------------
# Intent label mappings (6 behavioural clusters, 20 labels)
# ---------------------------------------------------------------------------
def get_intent_labels() -> list[str]:
    """Read intent labels from config.yaml across all 6 clusters."""
    cfg = load_config()
    labels = []
    for cluster in (
        "neutral_benign", "social_bonding", "passive_aggression",
        "active_aggression", "manipulation", "evasion",
    ):
        labels.extend(cfg["intents"][cluster])
    return labels


INTENT_LABELS = get_intent_labels()
INTENT2IDX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
IDX2INTENT = {idx: label for label, idx in INTENT2IDX.items()}


class ModerationDataset(Dataset):
    """
    Unified dataset for toxicity + intent training.
    Expects a DataFrame with columns: text, toxic (0/1), intent (label string).

    Returns variable-length token sequences (no padding).
    Use ModerationCollator with DataLoader for dynamic batch padding.
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
            normalize_text(row["text"]),
            max_length=self.max_length,
            truncation=True,
            padding=False,          # No padding — collator handles it per-batch
            return_tensors=None,    # Returns plain lists, not tensors
        )
        item = {
            "input_ids": enc["input_ids"],           # list[int], variable length
            "attention_mask": enc["attention_mask"],  # list[int], variable length
        }
        if self.has_labels:
            item["toxicity_labels"] = torch.tensor(float(row.get("toxic", 0)))
            # has_intent=False → -100 so CrossEntropyLoss(ignore_index=-100) skips it
            has_intent = bool(row.get("has_intent", True))
            if has_intent:
                item["intent_labels"] = torch.tensor(
                    INTENT2IDX.get(row.get("intent", "question"), INTENT2IDX["question"]),
                    dtype=torch.long,
                )
            else:
                item["intent_labels"] = torch.tensor(-100, dtype=torch.long)
        return item


class ModerationCollator:
    """Dynamic padding — pads each batch to its longest sequence, not max_length.

    Why this matters:
      Average chat message is ~30-80 tokens, but max_length=256.
      Fixed padding wastes 70-90% of tokens as zeros.
      Attention is O(n^2) so halving seq length = 4x attention speedup.
      Empirically gives 2-3x training throughput improvement.

    Usage:
      collator = ModerationCollator(tokenizer)
      DataLoader(dataset, collate_fn=collator, ...)
    """

    def __init__(self, tokenizer, has_labels: bool = True):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.has_labels = has_labels

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_id] * pad_len)
            attention_masks.append(item["attention_mask"] + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        if self.has_labels and "toxicity_labels" in batch[0]:
            result["toxicity_labels"] = torch.stack(
                [item["toxicity_labels"] for item in batch]
            )
            result["intent_labels"] = torch.stack(
                [item["intent_labels"] for item in batch]
            )

        return result


# ===========================================================================
# Dataset Loaders — Phase 1 (Toxicity)
# ===========================================================================

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
        csv = path / "train.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["toxic"] = df.get("label", df.get("HS", 0)).astype(int)
            return df[["text", "toxic"]]
    return pd.DataFrame(columns=["text", "toxic"])


# ===========================================================================
# Dataset Loaders — Phase 2 (Toxicity — New Sources)
# ===========================================================================

def load_metahate(path: Path, sample: Optional[int] = 500_000) -> pd.DataFrame:
    """
    Load MetaHate (1.7M rows, TSV: label + text).
    Harmonizes 36 social-media hate speech datasets.
    """
    tsv = path / "available_metahate.tsv"
    if not tsv.exists():
        return pd.DataFrame(columns=["text", "toxic"])

    df = pd.read_csv(tsv, sep="\t", on_bad_lines="skip", engine="python")
    # Columns: label (0/1), text
    df = df.rename(columns={"label": "toxic"})
    df["toxic"] = df["toxic"].astype(int)
    df = df[["text", "toxic"]].dropna(subset=["text"])

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)

    return df


def load_toxigen(path: Path) -> pd.DataFrame:
    """
    Load ToxiGen (implicit/adversarial toxicity).
    Used in Stage 2 curriculum for sarcasm/microaggression hardening.
    """
    train_csv = path / "annotated_train.csv"
    if not train_csv.exists():
        return pd.DataFrame(columns=["text", "toxic"])

    df = pd.read_csv(train_csv)
    # label column: "hate" / "neutral"
    df["toxic"] = (df["label"] == "hate").astype(int)
    # Clean the text — ToxiGen has b'...' byte-string prefixes
    df["text"] = df["text"].str.replace(r"^b'|'$", "", regex=True)
    df = df[["text", "toxic"]].dropna(subset=["text"])

    # Also load test set (if it has a 'label' column)
    test_csv = path / "annotated_test.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        if "label" in df_test.columns:
            df_test["toxic"] = (df_test["label"] == "hate").astype(int)
            df_test["text"] = df_test["text"].str.replace(r"^b'|'$", "", regex=True)
            df_test = df_test[["text", "toxic"]].dropna(subset=["text"])
            df = pd.concat([df, df_test], ignore_index=True)
        elif "toxicity_human" in df_test.columns:
            # Fallback: use human toxicity score (1-5 scale, >= 3 = toxic)
            df_test["toxic"] = (df_test["toxicity_human"] >= 3).astype(int)
            df_test["text"] = df_test["text"].str.replace(r"^b'|'$", "", regex=True)
            df_test = df_test[["text", "toxic"]].dropna(subset=["text"])
            df = pd.concat([df, df_test], ignore_index=True)

    return df


def load_synthetic_intents(path: Path) -> pd.DataFrame:
    """
    Load LLM-generated synthetic utterances for the 17 missing intent classes.
    Generated by scripts/generate_synthetic_data.py using Groq Llama 3.3 70B.
    Grounded in 2025-2026 research on grooming, radicalization, and gaming toxicity.
    All rows have has_intent=True — real intent labels, not defaults.
    """
    csv_file = path / "synthetic_utterances.csv"
    if not csv_file.exists():
        return pd.DataFrame(columns=["text", "toxic", "intent", "has_intent"])

    df = pd.read_csv(csv_file)
    df["has_intent"] = True
    df["toxic"] = df["toxic"].astype(int)
    return df[["text", "toxic", "intent", "has_intent"]].dropna(subset=["text"])


def load_toxic_chat(path: Path) -> pd.DataFrame:
    """
    Load Toxic-Chat dataset (lmsys/toxic-chat).
    5,082 user→ChatGPT conversation logs with human toxicity + jailbreaking labels.

    Why this matters:
    - ONLY chat-domain data in our pipeline (everything else is Wikipedia/news/scripted).
    - `jailbreaking=1` rows are mapped to intent=social_engineering with has_intent=True.
      These are real examples of manipulation/social engineering in natural chat language.
    - `jailbreaking=0, toxicity=1` rows add toxicity signal without intent label.
    - `jailbreaking=0, toxicity=0` rows add clean chat-domain negatives.

    Column mapping:
      user_input  → text
      toxicity    → toxic  (0/1, human annotated)
      jailbreaking→ intent=social_engineering when 1, else has_intent=False
    """
    # Try 1: HuggingFace arrow format (from download_datasets.py → save_to_disk)
    dfs = []
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        for split_name in ds:
            dfs.append(ds[split_name].to_pandas())
    except Exception:
        pass

    # Try 2: raw CSVs (manually downloaded)
    if not dfs:
        for split in ("toxic-chat_annotation_train.csv", "toxic-chat_annotation_test.csv"):
            csv_file = path / split
            if csv_file.exists():
                dfs.append(pd.read_csv(csv_file))

    if not dfs:
        return pd.DataFrame(columns=["text", "toxic", "intent", "has_intent"])

    rows = []
    for df in dfs:
        if "user_input" not in df.columns:
            continue

        # Binarise toxicity — stored as 0/1 int or float
        df["toxic"] = pd.to_numeric(df["toxicity"], errors="coerce").fillna(0).astype(int)
        df["text"] = df["user_input"].astype(str)

        # Jailbreaking rows → social_engineering intent label (chat-domain gold signal)
        if "jailbreaking" in df.columns:
            jb_col = pd.to_numeric(df["jailbreaking"], errors="coerce").fillna(0).astype(int)
        else:
            jb_col = pd.Series(0, index=df.index)
        df["is_jailbreak"] = jb_col.astype(bool)

        # has_intent=True only for jailbreaking rows (clear social_engineering signal)
        df["has_intent"] = df["is_jailbreak"]
        df["intent"] = df["is_jailbreak"].map({True: "social_engineering", False: "question"})

        rows.append(df[["text", "toxic", "intent", "has_intent"]].dropna(subset=["text"]))

    if not rows:
        return pd.DataFrame(columns=["text", "toxic", "intent", "has_intent"])
    return pd.concat(rows, ignore_index=True)


def load_session_data(path: Path) -> pd.DataFrame:
    """
    Load session sequences for HMM + classifier training.
    Generated by scripts/generate_synthetic_data.py.
    Each row is one turn in a multi-turn conversation (session_id + turn_id).
    """
    all_dfs = []
    for csv_file in path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if "text" in df.columns and "intent" in df.columns:
            df["toxic"] = df.get("toxic", 0).fillna(0).astype(int)
            df["has_intent"] = True
            all_dfs.append(df[["text", "toxic", "intent", "has_intent"]])
    if not all_dfs:
        return pd.DataFrame(columns=["text", "toxic", "intent", "has_intent"])
    return pd.concat(all_dfs, ignore_index=True).dropna(subset=["text"])


def load_civil_comments(path: Path, sample: Optional[int] = 500_000) -> pd.DataFrame:
    """
    Load Civil Comments (Parquet, continuous toxicity score).
    Large-scale diverse domain data for Stage 1 foundation.
    """
    parquet_files = sorted(path.glob("train-*.parquet"))
    if not parquet_files:
        return pd.DataFrame(columns=["text", "toxic"])

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    # Binarize toxicity score at 0.5 threshold (same as Jigsaw)
    df["toxic"] = (df["toxicity"] >= 0.5).astype(int)
    df = df[["text", "toxic"]].dropna(subset=["text"])

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)

    return df


# ===========================================================================
# Dataset Loaders — Phase 2 (Intent Pre-training)
# ===========================================================================

# Mapping from Banking77/CLINC150 labels → our 20-label taxonomy
BANKING77_INTENT_MAP = {
    # Most banking queries map to "question" or "information_sharing"
    "default": "question",
}

CLINC150_INTENT_MAP = {
    # CLINC150 benign intents → our benign cluster
    "greeting": "greeting",
    "goodbye": "greeting",         # farewell maps to greeting cluster
    "thanks": "feedback",
    "yes": "small_talk",
    "no": "small_talk",
    "oos": "question",             # out-of-scope → question (safe default)
    "default": "question",
}

DAILY_DIALOG_INTENT_MAP = {
    # DeepPavlov/daily_dialog act_label_text values
    "inform": "information_sharing",
    "question": "question",
    "directive": "feedback",
    "commissive": "solidarity_seeking",
}


def load_banking77(path: Path) -> pd.DataFrame:
    """
    Load Banking77 (13k customer service intents).
    Teaches geometric separation of similar utterances.
    All samples are benign (toxic=0).
    """
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        df = ds["train"].to_pandas()
    except Exception:
        csv = path / "train.csv"
        if csv.exists():
            df = pd.read_csv(csv)
        else:
            return pd.DataFrame(columns=["text", "toxic", "intent"])

    df["toxic"] = 0
    df["intent"] = df.get("label", "").apply(
        lambda x: BANKING77_INTENT_MAP.get(str(x), "question")
    )
    if "text" not in df.columns:
        # Banking77 uses 'text' or 'label' columns depending on source
        text_col = [c for c in df.columns if c not in ("label", "toxic", "intent")][0]
        df = df.rename(columns={text_col: "text"})

    df["has_intent"] = True
    return df[["text", "toxic", "intent", "has_intent"]].dropna(subset=["text"])


def load_clinc150(path: Path) -> pd.DataFrame:
    """
    Load CLINC150 (23k intent classification, 150 intents + OOS).
    Filtered to benign subset only — teaches OOS detection.
    All samples are benign (toxic=0).
    """
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        df = ds["train"].to_pandas()
    except Exception:
        csv = path / "train.csv"
        if csv.exists():
            df = pd.read_csv(csv)
        else:
            return pd.DataFrame(columns=["text", "toxic", "intent"])

    df["toxic"] = 0
    # Map CLINC150 intent names to our taxonomy
    if "intent" in df.columns:
        df["intent"] = df["intent"].apply(
            lambda x: CLINC150_INTENT_MAP.get(str(x), "question")
        )
    else:
        df["intent"] = "question"

    if "text" not in df.columns:
        text_col = [c for c in df.columns if c not in ("intent", "label", "toxic")][0]
        df = df.rename(columns={text_col: "text"})

    df["has_intent"] = True
    return df[["text", "toxic", "intent", "has_intent"]].dropna(subset=["text"])


def load_daily_dialog(path: Path) -> pd.DataFrame:
    """
    Load DailyDialog (87k utterances from DeepPavlov/daily_dialog).
    Domain Bridge: transitions from formal to informal register.
    Columns: dialog (list[str]), act_label_text, emotion_label_text.
    All samples are benign (toxic=0).
    """
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        df = ds["train"].to_pandas()
    except Exception:
        return pd.DataFrame(columns=["text", "toxic", "intent"])

    # Each row has dialog (list of utterances) and act_label_text
    # Flatten: one row per utterance
    rows = []
    for _, row in df.iterrows():
        utterances = row["dialog"] if isinstance(row["dialog"], list) else [row["dialog"]]
        act = row.get("act_label_text", "inform")
        intent = DAILY_DIALOG_INTENT_MAP.get(act, "small_talk")
        for utt in utterances:
            text = str(utt).strip()
            if text:
                rows.append({"text": text, "toxic": 0, "intent": intent})

    df_out = pd.DataFrame(rows)[["text", "toxic", "intent"]].dropna(subset=["text"])
    df_out["has_intent"] = True
    return df_out


# ===========================================================================
# Harmonization: Assign intent labels to toxicity-only datasets
# ===========================================================================

def assign_default_intents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign intent labels and has_intent flag after concatenation.
    Rows with has_intent=False get intent_labels=-100 at collation time,
    so CrossEntropyLoss(ignore_index=-100) skips them entirely.
    Only Banking77 / CLINC150 / DailyDialog rows have has_intent=True.
    """
    # Fill has_intent: rows from intent-labeled loaders have True; rest NaN → False
    if "has_intent" not in df.columns:
        df["has_intent"] = False
    else:
        df["has_intent"] = df["has_intent"].fillna(False).astype(bool)

    # Fill missing intent strings (NaN from toxicity-only rows) with a placeholder
    # These won't affect training because has_intent=False → label=-100
    if "intent" not in df.columns:
        df["intent"] = "question"
    else:
        mask = df["intent"].isna()
        df.loc[mask & (df["toxic"] == 1), "intent"] = "direct_attack"
        df.loc[mask & (df["toxic"] == 0), "intent"] = "question"
    return df


# ===========================================================================
# Build Dataset (Phase 2 — Curriculum-Aware)
# ===========================================================================

def build_train_dataset(
    tokenizer: AutoTokenizer,
    data_dir: Path = Path("data/raw"),
    max_length: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stage: int = 1,   # 1 = Explicit Foundation, 2 = add Implicit/Adversarial
) -> tuple[ModerationDataset, ModerationDataset, ModerationDataset]:
    """
    Load, merge, split, return (train, val, test) datasets.

    Stage 1: MetaHate + Jigsaw 2018/2019 + Civil Comments + Intent pre-training
    Stage 2: Stage 1 + ToxiGen + Session Data (implicit/adversarial hardening)
    """
    dfs = []

    # --- Stage 1: Explicit Foundation ---

    # Toxicity datasets
    jigsaw_2018 = data_dir / "jigsaw_2018"
    if (jigsaw_2018 / "train.csv").exists():
        df = load_jigsaw_2018(jigsaw_2018)
        df["source"] = "jigsaw_2018"
        dfs.append(df)
        print(f"  Loaded Jigsaw 2018: {len(df):,} rows")

    jigsaw_2019 = data_dir / "jigsaw_2019"
    if (jigsaw_2019 / "train.csv").exists():
        df = load_jigsaw_2019(jigsaw_2019)
        df["source"] = "jigsaw_2019"
        dfs.append(df)
        print(f"  Loaded Jigsaw 2019: {len(df):,} rows")

    hateval = data_dir / "hateval"
    if hateval.exists():
        df = load_hateval(hateval)
        if len(df) > 0:
            df["source"] = "hateval"
            dfs.append(df)
            print(f"  Loaded HatEval: {len(df):,} rows")

    metahate = data_dir / "metahate"
    if metahate.exists():
        df = load_metahate(metahate)
        if len(df) > 0:
            df["source"] = "metahate"
            dfs.append(df)
            print(f"  Loaded MetaHate: {len(df):,} rows")

    civil = data_dir / "civil_comments"
    if civil.exists():
        df = load_civil_comments(civil)
        if len(df) > 0:
            df["source"] = "civil_comments"
            dfs.append(df)
            print(f"  Loaded Civil Comments: {len(df):,} rows")

    # Synthetic intent data (17 missing classes, LLM-generated, all has_intent=True)
    synth = data_dir / "synthetic_intents"
    if synth.exists():
        df = load_synthetic_intents(synth)
        if len(df) > 0:
            df["source"] = "synthetic_intents"
            dfs.append(df)
            print(f"  Loaded Synthetic Intents: {len(df):,} rows ({df['intent'].nunique()} classes)")

    # Session sequences (for classifier + HMM, has_intent=True)
    session_dir = data_dir / "session_data"
    if session_dir.exists():
        df = load_session_data(session_dir)
        if len(df) > 0:
            df["source"] = "session_data"
            dfs.append(df)
            print(f"  Loaded Session Data: {len(df):,} rows")

    # Toxic-Chat: only chat-domain dataset in pipeline.
    # Jailbreaking rows → social_engineering intent label (has_intent=True).
    # All other rows → toxicity signal only (has_intent=False).
    toxic_chat_dir = data_dir / "toxic_chat"
    if toxic_chat_dir.exists():
        df = load_toxic_chat(toxic_chat_dir)
        if len(df) > 0:
            jb_rows = df["has_intent"].sum()
            df["source"] = "toxic_chat"
            dfs.append(df)
            print(f"  Loaded Toxic-Chat: {len(df):,} rows  "
                  f"({jb_rows} jailbreaking → social_engineering intent labels)")

    # Intent pre-training datasets (all benign, carry their own intent labels)
    banking = data_dir / "banking77"
    if banking.exists():
        df = load_banking77(banking)
        if len(df) > 0:
            df["source"] = "banking77"
            dfs.append(df)
            print(f"  Loaded Banking77: {len(df):,} rows")

    clinc = data_dir / "clinc150"
    if clinc.exists():
        df = load_clinc150(clinc)
        if len(df) > 0:
            df["source"] = "clinc150"
            dfs.append(df)
            print(f"  Loaded CLINC150: {len(df):,} rows")

    daily = data_dir / "daily_dialog"
    if daily.exists():
        df = load_daily_dialog(daily)
        if len(df) > 0:
            df["source"] = "daily_dialog"
            dfs.append(df)
            print(f"  Loaded DailyDialog: {len(df):,} rows")

    # --- Stage 2: Implicit & Adversarial ---
    if stage >= 2:
        toxigen = data_dir / "toxigen"
        if toxigen.exists():
            df = load_toxigen(toxigen)
            if len(df) > 0:
                df["source"] = "toxigen"
                dfs.append(df)
                print(f"  Loaded ToxiGen (Stage 2): {len(df):,} rows")

        # Session data now loaded in Stage 1 (above) — skip here to avoid duplication

    if not dfs:
        raise FileNotFoundError(
            "No datasets found in data/raw/. Run: python scripts/download_datasets.py"
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Assign default intents to toxicity-only rows
    combined = assign_default_intents(combined)

    # Validate intent labels — warn about unmapped labels
    valid_intents = set(INTENT2IDX.keys())
    unknown = set(combined["intent"].unique()) - valid_intents
    if unknown:
        print(f"  WARNING: {len(unknown)} unknown intent labels found, defaulting to 'question': {unknown}")
        combined.loc[~combined["intent"].isin(valid_intents), "intent"] = "question"

    # Stratified split
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(
        combined, test_size=test_ratio, stratify=combined["toxic"], random_state=seed
    )
    val_size = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["toxic"], random_state=seed
    )

    print(f"\n  Split: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    print(f"  Toxic ratio — train: {train['toxic'].mean():.3f} | val: {val['toxic'].mean():.3f}")

    # Intent distribution summary + class weight computation
    intent_train = train[train["has_intent"] == True]
    print(f"\n  Intent distribution (train, has_intent=True: {len(intent_train):,} rows):")
    for intent, count in intent_train["intent"].value_counts().head(10).items():
        print(f"    {intent}: {count:,} ({100*count/len(intent_train):.1f}%)")
    if len(intent_train["intent"].unique()) > 10:
        print(f"    ... and {len(intent_train['intent'].unique()) - 10} more labels")

    # Inverse-frequency class weights for CrossEntropyLoss.
    # Corrects the 1154:1 imbalance between "question" (54k rows) and rare
    # malicious intent classes (~47-70 rows). weight_i = N / (C * count_i).
    counts = intent_train["intent"].value_counts()
    n_total = len(intent_train)
    n_classes = len(INTENT_LABELS)
    weights = torch.zeros(n_classes)
    for label, idx in INTENT2IDX.items():
        count = counts.get(label, 1)   # default to 1 to avoid division by zero
        weights[idx] = n_total / (n_classes * count)
    # Clip extreme weights: cap at 50× median to prevent instability on ultra-rare classes
    median_w = weights.median()
    weights = weights.clamp(max=50.0 * median_w)
    print(f"\n  Intent class weights (inverse-freq, clipped at 50x median):")
    top3_heavy = sorted(INTENT_LABELS, key=lambda l: -weights[INTENT2IDX[l]])[:3]
    top3_light = sorted(INTENT_LABELS, key=lambda l: weights[INTENT2IDX[l]])[:3]
    for l in top3_heavy:
        print(f"    {l:<25}  w={weights[INTENT2IDX[l]]:.2f}  (rare  → upweighted)")
    for l in top3_light:
        print(f"    {l:<25}  w={weights[INTENT2IDX[l]]:.2f}  (common → downweighted)")

    return (
        ModerationDataset(train, tokenizer, max_length),
        ModerationDataset(val, tokenizer, max_length),
        ModerationDataset(test, tokenizer, max_length),
        weights,   # (n_intents,) float tensor — pass to ContentModerationModel
    )
