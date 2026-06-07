"""
mine_hard_negatives.py — Extract high-confidence wrong predictions from Stage 2 val set.

Hard negatives = examples where the model is confidently wrong.
These are the highest-value training signal for Stage 3.

Usage:
    python scripts/mine_hard_negatives.py
    python scripts/mine_hard_negatives.py --max-samples 100   # quick test
    python scripts/mine_hard_negatives.py --max-samples 50000 # full val set (Kaggle)
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel

from src.classifier.model import ContentModerationModel
from src.classifier.dataset import (
    build_train_dataset, ModerationCollator, INTENT_LABELS, IDX2INTENT
)

import os
from dotenv import load_dotenv

load_dotenv()

TOX_THRESHOLD = float(os.getenv("TOX_THRESHOLD", 0.54))


def load_model(checkpoint: str, device: str):
    ckpt = Path(checkpoint)
    print(f"Loading checkpoint: {ckpt.resolve()}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = ContentModerationModel(
        encoder_name="answerdotai/ModernBERT-large",
        n_intents=len(INTENT_LABELS),
        focal_alpha=0.75, focal_gamma=1.0,
        alpha=0.65, beta=0.5, label_smoothing=0.1,
    )
    model.encoder = PeftModel.from_pretrained(
        model.encoder, str(ckpt)
    ).merge_and_unload()
    heads = torch.load(ckpt / "heads.pt", map_location="cpu", weights_only=True)
    model.toxicity_head.load_state_dict(heads["toxicity_head"])
    model.intent_head.load_state_dict(heads["intent_head"])
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def get_val_predictions(model, tokenizer, stage: int, max_samples: int,
                        device: str, batch_size: int = 64):
    print(f"Loading val data (stage={stage}, max_samples={max_samples})...")
    _, val_ds, _, _ = build_train_dataset(
        tokenizer,
        max_length=256,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        stage=stage,
    )

    if max_samples and max_samples < len(val_ds):
        indices = np.random.RandomState(42).choice(
            len(val_ds), max_samples, replace=False
        )
        val_ds = torch.utils.data.Subset(val_ds, indices)

    print(f"  Val samples to score: {len(val_ds):,}")

    collator = ModerationCollator(tokenizer, has_labels=True)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)

    all_scores, all_labels, all_intent_preds, all_texts = [], [], [], []

    for i, batch in enumerate(loader):
        if i % 20 == 0:
            print(f"  Batch {i}/{len(loader)}...")
        out = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        scores = out.toxicity_score.cpu().float().numpy()
        intent_idxs = out.intent_logits.argmax(-1).cpu().numpy()
        labels = batch["toxicity_labels"].cpu().numpy()

        all_scores.extend(scores)
        all_labels.extend(labels)
        all_intent_preds.extend([IDX2INTENT.get(int(idx), "question") for idx in intent_idxs])

        # Recover original texts from token IDs (best effort decode)
        for ids in batch["input_ids"]:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            all_texts.append(text)

    return (
        np.array(all_scores),
        np.array(all_labels),
        all_intent_preds,
        all_texts,
    )


def mine_hard_negatives(
    scores, labels, intent_preds, texts,
    tox_threshold: float,
    fp_conf_threshold: float = 0.65,
    fn_conf_threshold: float = 0.35,
):
    """
    Identify hard negatives:
    - False positives (FP): model score > tox_threshold but true label = 0, AND score > fp_conf_threshold
    - False negatives (FN): model score < tox_threshold but true label = 1, AND score < fn_conf_threshold
    """
    rows = []
    for text, score, label, intent in zip(texts, scores, labels, intent_preds):
        is_fp = (score > tox_threshold) and (label == 0) and (score > fp_conf_threshold)
        is_fn = (score < tox_threshold) and (label == 1) and (score < fn_conf_threshold)

        if is_fp:
            rows.append({
                "text": text,
                "true_label": int(label),
                "predicted_score": float(score),
                "error_type": "false_positive",
                "intent_label": intent,
            })
        elif is_fn:
            rows.append({
                "text": text,
                "true_label": int(label),
                "predicted_score": float(score),
                "error_type": "false_negative",
                "intent_label": intent,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/best_multitask")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Val samples to score (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp-threshold", type=float, default=0.65,
                        help="Min score for FP (high confidence wrong positive)")
    parser.add_argument("--fn-threshold", type=float, default=0.35,
                        help="Max score for FN (high confidence wrong negative)")
    parser.add_argument("--output", default="data/raw/hard_negatives/hard_negatives.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tox threshold: {TOX_THRESHOLD}")
    print(f"FP confidence threshold: {args.fp_threshold}")
    print(f"FN confidence threshold: {args.fn_threshold}")

    model, tokenizer = load_model(args.checkpoint, device)

    scores, labels, intent_preds, texts = get_val_predictions(
        model, tokenizer, args.stage,
        args.max_samples, device, args.batch_size
    )

    print(f"\nMining hard negatives (threshold={TOX_THRESHOLD})...")
    df = mine_hard_negatives(
        scores, labels, intent_preds, texts,
        tox_threshold=TOX_THRESHOLD,
        fp_conf_threshold=args.fp_threshold,
        fn_conf_threshold=args.fn_threshold,
    )

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary
    fp_count = (df["error_type"] == "false_positive").sum()
    fn_count = (df["error_type"] == "false_negative").sum()

    print("\n" + "=" * 50)
    print(f"  Hard Negatives Summary")
    print(f"  Scored samples:     {len(scores):,}")
    print(f"  False positives:    {fp_count:,}  (score > {args.fp_threshold} but true=0)")
    print(f"  False negatives:    {fn_count:,}  (score < {args.fn_threshold} but true=1)")
    print(f"  Total saved:        {len(df):,}")
    print(f"  Output file:        {out_path}")
    print("=" * 50)

    # Verify required columns
    assert set(df.columns) >= {"text", "true_label", "predicted_score", "error_type", "intent_label"}, \
        "Output CSV missing required columns!"
    if len(df) > 0:
        assert df["error_type"].nunique() > 0, "No error types found!"
    print("\nColumn check: OK")
    return df


if __name__ == "__main__":
    main()
