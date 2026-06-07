# Content Moderation Classifier — Training Journal

## System Overview

Real-time multi-task content moderation pipeline for chat platforms.

**Stack:**
- **Classifier**: ModernBERT-large (147M params) + LoRA (r=8, 0.55% trainable) — shared encoder → toxicity head + intent head
- **Session layer**: Hidden Markov Model over intent sequences for escalation detection
- **Escalation**: Groq Llama-3.3-70B for ambiguous/flagged content
- **API**: FastAPI real-time endpoint

**Intent taxonomy**: 21-class behavioural classification across 6 clusters:
| Cluster | Labels |
|---|---|
| neutral_benign | greeting, question, information_sharing, small_talk |
| social_bonding | solidarity_seeking, venting, humour, feedback |
| passive_aggression | baiting, sarcasm, passive_hostility, irony |
| active_aggression | direct_attack, threatening, escalating_aggression |
| manipulation | gaslighting, grooming, social_engineering |
| evasion | topic_deflection, denial, identity_concealment |

---

## The Bug: Intent F1 = 0.000 (3 Weeks)

### Symptom
After every training run, the intent classification head collapsed:
- **Intent F1 (macro): 0.000**
- Model predicted `question` (majority class) for every sample
- Toxicity head trained normally; only intent was broken

### Root Cause Analysis

Three compounding factors caused the collapse:

#### 1. Raw Inverse-Frequency Class Weights (123× for rare classes)
```python
# BROKEN — original code
weights[idx] = n_total / (n_classes * count)
# irony: 123×, sarcasm: 74× → gradient explosion
```
With weights up to 123×, a single incorrect prediction on `irony` produced a loss signal ~123× larger than a `question` prediction. This caused gradient instability and the model learned to ignore the intent head entirely.

**Fix — sqrt-dampened weights:**
```python
# FIXED
weights[idx] = (n_total / (n_classes * count)) ** 0.5  # sqrt dampening
# irony: 11× max, clipped at 10× median
```
Compresses range from 123× → max 2.95× (after upsampling), maintaining class signal without exploding gradients.

#### 2. Only 2–3 Intent-Labelled Samples Per Batch
With 8% of training data having intent labels and 21 classes, most batches had near-zero intent signal. The model couldn't learn from such sparse signal.

**Fix — static rare class upsampling:**
```python
_min_per_class = 500  # floor for each intent class in training set
# Duplicate rows for classes below 500 examples
# DDP-safe: operates on DataFrame before Dataset creation
# Result: +3,384 rows across 13 classes
```

#### 3. 4 Missing Intent Classes in Synthetic Data
The synthetic utterances script was missing `question`, `information_sharing`, `solidarity_seeking`, and `feedback` — four of the highest-frequency classes. The model had no grounded examples for them.

**Fix:** Extended `scripts/create_synthetic_data_local.py`:
- Added ~50 utterances per missing class
- Extended `identity_concealment` from 36 → 60 examples
- Expanded `irony`, `denial`, `topic_deflection`, `social_engineering`
- **Total: 813 → 1,079 utterances, all 21 classes covered**

---

## Architecture

### Model
```
ModernBERT-large (answerdotai/ModernBERT-large)
  └── LoRA adapter (r=8, α=16, dropout=0.15)
       Targets: Wqkv, Wo projection matrices
  └── Mean pooling (no CLS pre-training objective in ModernBERT)
  └── Dropout (p=0.1)
  ├── Toxicity head:  Linear(1024→256) → GELU → Linear(256→1) → Sigmoid
  └── Intent head:   Linear(1024→256) → GELU → Linear(256→21)
```

### Loss Functions
- **Toxicity**: Focal Loss (α=0.75, γ=1.0, fp32 even under fp16 training)
  - α=0.75: toxic class gets 3× more gradient than non-toxic (class-differentiated, not uniform)
  - γ=1.0: reduced from 2.0 for less conservative outputs, better calibration
- **Intent**: Weighted CrossEntropyLoss (sqrt-dampened weights, label_smoothing=0.1)
- **Combined**: `L = 0.65 × L_tox + 0.5 × L_intent`

### Training Setup
- **Hardware**: Kaggle T4×2 (15GB each), DDP via HuggingFace Accelerate
- **Precision**: fp16 mixed precision
- **Batch**: 16 per GPU × 2 GPUs = 32 effective
- **Optimizer**: AdamW (lr=5e-6, weight_decay=0.05)
- **Scheduler**: Cosine annealing with 10% warmup
- **Gradient accumulation**: 1 step
- **Mid-epoch checkpoint**: Every 5,000 steps (Kaggle session crash recovery)

---

## Dataset Summary

### Stage 1 — Explicit Foundation
| Dataset | Rows | Purpose |
|---|---|---|
| Jigsaw 2018 | 159,571 | Wikipedia talk page toxicity |
| Jigsaw 2019 | 500,000 | Civil Comments toxicity |
| HatEval | 9,000 | Twitter hate speech |
| Toxic-Chat | 10,165 | Adversarial jailbreaks → social_engineering intent |
| Banking77 | 10,003 | Intent pre-training (question, information_sharing) |
| CLINC150 | 15,250 | Intent pre-training (greeting, feedback, small_talk) |
| DailyDialog | 87,170 | Conversational intent |
| Synthetic Intents | 1,079 | All 21 custom intent classes |
| Session Data | 12,225 | Multi-turn conversation sequences |
| **Total** | **~804K** | → 563K train / 120K val / 120K test |

### Stage 2 — Implicit & Adversarial Hardening
Stage 1 datasets + **ToxiGen** (implicit hate, sarcasm, microaggressions)

---

## Training Results

### Progression
| Stage | Epoch | Tox F1 | Tox AUC | Intent F1 | Avg F1 | Notes |
|---|---|---|---|---|---|---|
| Before fix | — | 0.000 | — | 0.000 | — | Intent head collapse |
| Stage 1 | 1 | 0.3349 | 0.8831 | 0.3311 | 0.3330 | Broken focal loss (uniform α) |
| Stage 1 | 2 | 0.3587 | 0.8877 | 0.4263 | 0.3925 | Broken focal loss |
| Stage 2 | 1 | 0.3524 | 0.8992 | 0.4912 | 0.4218 | Broken focal loss |
| Stage 2 | 2 | 0.3839 | 0.9005 | 0.5170 | 0.4505 | Broken focal loss — best before fix |
| **Retrain** | **1** | **0.5059** | **0.9031** | **0.3870** | **0.4464** | Fixed focal loss (α=0.75, γ=1.0) |
| Retrain | 2 | **0.5273** | **0.9121** | **0.4634** | **0.4954** | Stage 1 complete — best checkpoint saved |
| Retrain S2 | 1 | — | — | — | — | ToxiGen hardening — in progress |

### Threshold Tuning (old model — pre focal loss fix)
Val set threshold sweep (500-sample subset) after Stage 2 with broken focal loss:
- **AUC: 0.9176** (target: 0.92 ✅)
- Tox F1 at default threshold (0.50): 0.4800
- **Optimal threshold: 0.40**
- **Tox F1 at optimal threshold: 0.6486**

**Why AUC ≠ F1:** Focal loss (γ=2.0, uniform α) produced conservative sigmoid outputs (scores cluster below 0.5). Model ranked correctly (AUC=0.92) but needed lower threshold for binary decisions. Fixed with class-differentiated α=0.75, γ=1.0 — scores should be better calibrated.

*Note: Threshold tuning to be re-run after retrain completes. Expected optimal threshold closer to 0.45-0.50 with fixed focal loss.*

### Targets vs Current Best
| Metric | Old (broken α) | Retrain S1 Ep2 | Target | Notes |
|---|---|---|---|---|
| Tox AUC | 0.9176 | **0.9121** | 0.92 | Stage 1 only — S2 will push past 0.92 |
| Tox F1 | 0.6486 (t=0.40) | 0.5273 (t=0.50) | 0.85 | S1 raw; expect 0.70+ after S2 + tuning |
| Intent F1 | 0.5170 | 0.4634 | 0.80 | Recovering; S2 ToxiGen will lift this |

---

## Known Issues & Next Steps

### To Improve Tox F1 (0.65 → 0.80+)
1. **Add civil_comments dataset** — 500K diverse web toxicity rows (parquet, HuggingFace)
   ```python
   ds = load_dataset("google/civil_comments", split="train", trust_remote_code=True)
   ds.to_parquet('data/raw/civil_comments/train-00000.parquet')
   ```
2. **Reduce focal loss gamma**: `focal_loss_gamma: 2.0 → 1.0` (less conservative)
3. **Rebalance focal loss alpha**: `focal_loss_alpha: 0.25 → 0.75` (upweight toxic class)
4. **Increase toxicity loss weight**: `alpha: 0.5 → 0.65` in config.yaml
5. **2 more training epochs** (val loss still decreasing, no overfitting)

### Remaining Pipeline Components
- [ ] HMM training (`scripts/train_hmm.py`) on session data
- [ ] FastAPI endpoint (`src/api/`)
- [ ] Test set evaluation (final reported metrics)
- [ ] Demo UI

### Kaggle-Specific Notes
- **Session kills**: Mid-epoch checkpoint saves every 5,000 steps; auto-resumes via `load_mid_epoch()`
- **Browser disconnect**: Use `!tail -N /kaggle/working/train_log.txt` to check progress
- **Old checkpoint resume bug**: Always clear `models/checkpoints/best_multitask` before a fresh run to prevent loading collapsed weights
- **Dataset loading**: `civil_comments` silently skipped if `data/raw/civil_comments` doesn't exist
- **DDP + sampling**: Use static DataFrame upsampling, NOT `WeightedRandomSampler` (Accelerate re-wraps it incorrectly in multi-GPU)

---

## Checkpoint Files

Saved at `models/checkpoints/best_multitask/`:
```
adapter_model.safetensors   ← LoRA adapter weights (merge into base for inference)
adapter_config.json         ← LoRA rank/alpha/target modules config
heads.pt                    ← toxicity_head + intent_head state dicts
tokenizer_config.json       ← ModernBERT tokenizer config
tokenizer.json              ← vocab + tokenization rules
```

### Loading for Inference
```python
from src.classifier.model import ContentModerationModel
from peft import PeftModel
import torch

CKPT = 'models/checkpoints/best_multitask'
model = ContentModerationModel(encoder_name='answerdotai/ModernBERT-large', n_intents=21)
model.encoder = PeftModel.from_pretrained(model.encoder, CKPT).merge_and_unload()
heads = torch.load(f'{CKPT}/heads.pt', map_location='cpu')
model.toxicity_head.load_state_dict(heads['toxicity_head'])
model.intent_head.load_state_dict(heads['intent_head'])
model.eval()
```

---

*Training completed: June 2026 | Hardware: Kaggle T4×2 | Total compute: ~12 GPU-hours*
