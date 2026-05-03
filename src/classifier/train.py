"""
ModernBERT fine-tuning with LoRA (fits in 6GB VRAM).
Supports multi-task (toxicity + intent) and single-task ablations.
"""
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, roc_auc_score
import yaml

from src.classifier.model import ContentModerationModel
from src.classifier.dataset import build_train_dataset, INTENT_LABELS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def apply_lora(model: ContentModerationModel, cfg: dict) -> ContentModerationModel:
    """Wrap encoder with LoRA — only trains ~0.5% of parameters."""
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg["training"]["lora_r"],
        lora_alpha=cfg["training"]["lora_alpha"],
        lora_dropout=cfg["training"]["lora_dropout"],
        target_modules=["Wqkv", "Wo"],
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_config)
    trainable, total = model.encoder.get_nb_trainable_parameters() if hasattr(
        model.encoder, "get_nb_trainable_parameters"
    ) else (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))
    print(f"LoRA: {trainable:,} trainable / {total:,} total params "
          f"({100 * trainable / total:.2f}%)")
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_tox_scores, all_tox_labels = [], []
    all_intent_preds, all_intent_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tox_labels = batch["toxicity_labels"].to(device)
        int_labels = batch["intent_labels"].to(device)

        out = model(input_ids, attention_mask, tox_labels, int_labels)
        total_loss += out.loss.item()

        all_tox_scores.extend(out.toxicity_score.cpu().numpy())
        all_tox_labels.extend(tox_labels.cpu().numpy())
        all_intent_preds.extend(out.intent_logits.argmax(-1).cpu().numpy())
        all_intent_labels.extend(int_labels.cpu().numpy())

    tox_preds = (np.array(all_tox_scores) >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader),
        "toxicity_f1": f1_score(all_tox_labels, tox_preds, zero_division=0),
        "toxicity_auc": roc_auc_score(all_tox_labels, all_tox_scores),
        "intent_f1_macro": f1_score(
            all_intent_labels, all_intent_preds, average="macro", zero_division=0
        ),
    }


def train(
    task: str = "multitask",       # "multitask" | "toxicity_only" | "intent_only"
    use_lora: bool = True,
    run_name: str | None = None,
    checkpoint_dir: Path = Path("models/checkpoints"),
):
    cfg = load_config()
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(cfg["training"]["seed"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokeniser ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])

    # --- Data ---
    n_intents = len(INTENT_LABELS)
    train_ds, val_ds, _ = build_train_dataset(
        tokenizer,
        max_length=cfg["model"]["max_length"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["training"]["seed"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False, num_workers=8, pin_memory=True
    )

    # --- Model ---
    alpha = cfg["training"]["alpha"] if task != "intent_only" else 0.0
    beta = cfg["training"]["beta"] if task != "toxicity_only" else 0.0

    model = ContentModerationModel(
        encoder_name=cfg["model"]["encoder"],
        n_intents=n_intents,
        alpha=alpha,
        beta=beta,
        focal_gamma=cfg["training"]["focal_loss_gamma"],
        focal_alpha=cfg["training"]["focal_loss_alpha"],
    )

    # --- Model & Resume Logic ---
    resume_path = checkpoint_dir / f"best_{task}"
    
    if not args.full_ft and resume_path.exists():
        print(f"Resuming from existing checkpoint: {resume_path}")
        from peft import PeftModel
        model.encoder = PeftModel.from_pretrained(model.encoder, str(resume_path))
        heads_data = torch.load(resume_path / "heads.pt", map_location=device, weights_only=True)
        model.toxicity_head.load_state_dict(heads_data["toxicity_head"])
        model.intent_head.load_state_dict(heads_data["intent_head"])
        print("✓ Checkpoint loaded successfully.")
    else:
        if use_lora and not args.full_ft:
            model = apply_lora(model, cfg)
        else:
            print("🚀 FULL FINE-TUNING ENABLED: Training all 150M+ parameters.")
            for param in model.parameters():
                param.requires_grad = True

    model = model.to(device)
    
    # torch.compile disabled for Windows stability.
    
    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"] if not args.full_ft else 5e-6, # Lower LR for Full FT
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- W&B ---
    if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
        wandb.init(project="content-moderation", name=run_name or task, config=cfg)

    # --- Training loop ---
    best_val_f1 = 0.0
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["hardware"]["fp16"] and device.type == "cuda"))

    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tox_labels = batch["toxicity_labels"].to(device)
            int_labels = batch["intent_labels"].to(device)

            with torch.amp.autocast('cuda', enabled=(cfg["hardware"]["fp16"] and device.type == "cuda")):
                out = model(input_ids, attention_mask, tox_labels, int_labels)
                loss = out.loss / grad_accum  # scale loss for accumulation

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum  # unscale for logging

            if step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                      f"Loss: {loss.item() * grad_accum:.4f}")

        # --- Validation ---
        val_metrics = evaluate(model, val_loader, device)
        avg_f1 = (val_metrics["toxicity_f1"] + val_metrics["intent_f1_macro"]) / 2

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train loss:   {total_loss / len(train_loader):.4f}")
        print(f"  Val loss:     {val_metrics['loss']:.4f}")
        print(f"  Tox F1:       {val_metrics['toxicity_f1']:.4f}")
        print(f"  Tox AUC:      {val_metrics['toxicity_auc']:.4f}")
        print(f"  Intent F1:    {val_metrics['intent_f1_macro']:.4f}")

        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"epoch": epoch + 1, **val_metrics,
                       "train_loss": total_loss / len(train_loader)})

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            save_path = checkpoint_dir / f"best_{task}"
            
            if hasattr(model.encoder, "save_pretrained"):
                model.encoder.save_pretrained(str(save_path))
            else:
                # Save full model if not using LoRA
                torch.save(model.encoder.state_dict(), save_path / "pytorch_model.bin")
                
            tokenizer.save_pretrained(str(save_path))
            torch.save({
                "toxicity_head": model.toxicity_head.state_dict(),
                "intent_head": model.intent_head.state_dict(),
                "config": cfg,
                "task": task,
            }, save_path / "heads.pt")
            print(f"  ✓ Saved best model (avg F1: {avg_f1:.4f}) → {save_path}")

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    print(f"\nTraining complete. Best val avg F1: {best_val_f1:.4f}")
    return str(checkpoint_dir / f"best_{task}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="multitask",
                        choices=["multitask", "toxicity_only", "intent_only"])
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--full-ft", action="store_true", help="Enable Full Fine-Tuning")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    train(task=args.task, use_lora=not args.no_lora, run_name=args.run_name)
