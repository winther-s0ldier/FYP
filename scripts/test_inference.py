import torch
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer

# Add the project root to sys.path so we can import our modules
sys.path.append(os.getcwd())

from src.classifier.model import ContentModerationModel
from src.classifier.dataset import INTENT_LABELS, IDX2INTENT

def run_inference():
    # 1. Setup paths
    checkpoint_path = Path("models/checkpoints/primitive_checkpoint")
    heads_path = checkpoint_path / "heads.pt"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"--- Loading Primitive Checkpoint ---")
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # 3. Initialize Model
    # Note: Using 'answerdotai/ModernBERT-large' as the base architecture
    # but loading weights from our local checkpoint.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ContentModerationModel(
        encoder_name="answerdotai/ModernBERT-large", 
        n_intents=len(INTENT_LABELS),
        attn_implementation="sdpa" if device == "cuda" else "eager"
    )
    
    # Load encoder weights
    # Note: ModernBERT weights are in the checkpoint folder
    from transformers import AutoModel
    model.encoder = AutoModel.from_pretrained(checkpoint_path)
    
    # Load custom heads
    checkpoint = torch.load(heads_path, map_location=device)
    model.toxicity_head.load_state_dict(checkpoint["toxicity_head"])
    model.intent_head.load_state_dict(checkpoint["intent_head"])
    
    model.to(device)
    model.eval()
    
    print("\n✅ Model Loaded Successfully!")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("\nEnter text to test: ")
        if text.lower() == 'exit':
            break
            
        if not text.strip():
            continue

        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # Predict
        with torch.no_grad():
            output = model.predict(inputs["input_ids"], inputs["attention_mask"])
            
        # Parse Results
        tox_score = output["toxicity_score"][0]
        intent_idx = output["intent_label_idx"][0]
        intent_label = IDX2INTENT[intent_idx]
        confidence = output["confidence"][0]
        
        # Color coding for terminal
        color = "\033[91m" if tox_score > 0.5 else "\033[92m" # Red if toxic, Green if clean
        reset = "\033[0m"
        
        print(f"\n--- Results ---")
        print(f"Text: \"{text}\"")
        print(f"Toxicity: {color}{tox_score:.4f}{reset} ({'TOXIC' if tox_score > 0.5 else 'CLEAN'})")
        print(f"Intent:   \033[94m{intent_label}\033[0m (Index: {intent_idx})")
        print(f"Confidence: {confidence:.4f}")
        print(f"----------------")

if __name__ == "__main__":
    run_inference()
