"""
ModernBERT fine-tuning with LoRA + HuggingFace Accelerate.
Supports multi-task (toxicity + intent), single-task ablations, and
Phase 2 curriculum training (--stage 1 / --stage 2).

Single-GPU:  python -m src.classifier.train --stage 1
Two GPUs:    accelerate launch --num_processes 2 -m src.classifier.train --stage 1
"""
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import f1_score, roc_auc_score
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
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
def evaluate(model, loader, accelerator: Accelerator) -> dict:
    model.eval()
    all_tox_scores, all_tox_labels = [], []
    all_intent_preds, all_intent_labels = [], []
    total_loss = 0.0

    for batch in loader:
        # Accelerate already moved batch tensors to the correct device
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        tox_labels = batch["toxicity_labels"]
        int_labels = batch["intent_labels"]

        out = model(input_ids, attention_mask, tox_labels, int_labels)
        total_loss += out.loss.item()

        # Gather across all processes before collecting
        tox_score_g = accelerator.gather_for_metrics(out.toxicity_score)
        tox_label_g = accelerator.gather_for_metrics(tox_labels)
        intent_pred_g = accelerator.gather_for_metrics(out.intent_logits.argmax(-1))
        intent_label_g = accelerator.gather_for_metrics(int_labels)

        all_tox_scores.extend(tox_score_g.cpu().numpy())
        all_tox_labels.extend(tox_label_g.cpu().numpy())
        all_intent_preds.extend(intent_pred_g.cpu().numpy())
        all_intent_labels.extend(intent_label_g.cpu().numpy())

    tox_preds = (np.array(all_tox_scores) >= 0.5).astype(int)

    # Filter out samples without real intent labels (intent_label == -100)
    intent_labels_arr = np.array(all_intent_labels)
    intent_preds_arr = np.array(all_intent_preds)
    intent_mask = intent_labels_arr != -100
    intent_f1 = f1_score(
        intent_labels_arr[intent_mask], intent_preds_arr[intent_mask],
        average="macro", zero_division=0,
    ) if intent_mask.any() else 0.0

    return {
        "loss": total_loss / len(loader),
        "toxicity_f1": f1_score(all_tox_labels, tox_preds, zero_division=0),
        "toxicity_auc": roc_auc_score(all_tox_labels, all_tox_scores),
        "intent_f1_macro": intent_f1,
    }


def train(
    task: str = "multitask",       # "multitask" | "toxicity_only" | "intent_only"
    use_lora: bool = True,
    full_ft: bool = False,         # Full fine-tuning (all params)
    stage: int = 1,                # Curriculum stage: 1=Explicit, 2=Implicit
    run_name: str | None = None,
    checkpoint_dir: Path = Path("models/checkpoints"),
):
    cfg = load_config()

    # --- Accelerate (handles device, mixed precision, DDP automatically) ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    mixed_precision = "fp16" if cfg["hardware"]["fp16"] else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Device: {device}  |  Num processes: {accelerator.num_processes}")
        print(f"Mixed precision: {mixed_precision}")
        print(f"Curriculum Stage: {stage}")
        print("⚡ PyTorch Native Flash Attention (SDPA) activated!")

    torch.manual_seed(cfg["training"]["seed"])

    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokeniser ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])

    # --- Data (curriculum-aware) ---
    n_intents = len(INTENT_LABELS)
    train_ds, val_ds, _ = build_train_dataset(
        tokenizer,
        max_length=cfg["model"]["max_length"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["training"]["seed"],
        stage=stage,
    )

    num_workers = cfg["hardware"].get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False, num_workers=num_workers, pin_memory=True
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
        attn_implementation="sdpa",
    )

    # --- Model & Resume Logic ---
    resume_path = checkpoint_dir / f"best_{task}"

    if not full_ft and resume_path.exists():
        if accelerator.is_main_process:
            print(f"Resuming from existing checkpoint: {resume_path}")
        model.encoder = PeftModel.from_pretrained(model.encoder, str(resume_path))
        heads_data = torch.load(resume_path / "heads.pt", map_location="cpu", weights_only=True)
        model.toxicity_head.load_state_dict(heads_data["toxicity_head"])
        model.intent_head.load_state_dict(heads_data["intent_head"])
        if accelerator.is_main_process:
            print("✓ Checkpoint loaded successfully.")
    else:
        if use_lora and not full_ft:
            model = apply_lora(model, cfg)
        else:
            if accelerator.is_main_process:
                print("🚀 FULL FINE-TUNING ENABLED: Training all 150M+ parameters.")
            for param in model.parameters():
                param.requires_grad = True

    # --- Optimiser ---
    lr = cfg["training"]["learning_rate"] if not full_ft else 5e-6
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Accelerate: prepare everything (DDP wrap + device move) ---
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # --- W&B (main process only) ---
    if accelerator.is_main_process and WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
        wandb.init(project="content-moderation", name=run_name or task, config=cfg)

    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)

    # --- Training loop ---
    best_val_f1 = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # accelerator.accumulate handles the grad_accum + sync logic
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                tox_labels = batch["toxicity_labels"]
                int_labels = batch["intent_labels"]

                out = model(input_ids, attention_mask, tox_labels, int_labels)
                loss = out.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().item()

            if step % 100 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                      f"Loss: {loss.detach().item():.4f}")

        # --- Validation (main process collects gathered metrics) ---
        val_metrics = evaluate(model, val_loader, accelerator)
        avg_f1 = (val_metrics["toxicity_f1"] + val_metrics["intent_f1_macro"]) / 2

        if accelerator.is_main_process:
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
                save_path.mkdir(parents=True, exist_ok=True)

                # Unwrap DDP wrapper before saving
                unwrapped = accelerator.unwrap_model(model)

                if hasattr(unwrapped.encoder, "save_pretrained"):
                    unwrapped.encoder.save_pretrained(str(save_path))
                else:
                    torch.save(unwrapped.encoder.state_dict(),
                               save_path / "pytorch_model.bin")

                tokenizer.save_pretrained(str(save_path))
                torch.save({
                    "toxicity_head": unwrapped.toxicity_head.state_dict(),
                    "intent_head": unwrapped.intent_head.state_dict(),
                    "config": cfg,
                    "task": task,
                }, save_path / "heads.pt")
                print(f"  ✓ Saved best model (avg F1: {avg_f1:.4f}) → {save_path}")

    if accelerator.is_main_process and WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\nTraining complete. Best val avg F1: {best_val_f1:.4f}")

    return str(checkpoint_dir / f"best_{task}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 Curriculum Training")
    parser.add_argument("--task", default="multitask",
                        choices=["multitask", "toxicity_only", "intent_only"])
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Curriculum stage: 1=Explicit Foundation, 2=Implicit/Adversarial")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--full-ft", action="store_true", help="Enable Full Fine-Tuning")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    train(
        task=args.task,
        use_lora=not args.no_lora,
        full_ft=args.full_ft,
        stage=args.stage,
        run_name=args.run_name,
    )
