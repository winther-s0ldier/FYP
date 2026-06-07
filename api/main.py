"""
FastAPI application — REST + WebSocket endpoints.
Mount: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Environment variables (set in .env):
  BASE_MODEL          Base encoder name (default: answerdotai/ModernBERT-large)
  MODEL_CHECKPOINT    Path to LoRA checkpoint dir (default: models/checkpoints/best_multitask)
  TOX_THRESHOLD       Binary classification threshold, tuned on val set (default: 0.40)
  REDIS_URL           Redis connection string (default: redis://localhost:6379)
  DATABASE_URL        PostgreSQL DSN (default: postgresql://...)
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

from src.classifier.model import ContentModerationModel
from src.classifier.dataset import INTENT_LABELS
from src.pipeline.trie import SlurTrie
from src.pipeline.pipeline import ModerationPipeline, ClassificationRequest
from src.session.store import SessionStore
from src.session.hmm import SessionHMM
from src.db.postgres import MessageStore
from api.routes.classify import router as classify_router
from api.routes.dashboard import router as dashboard_router

load_dotenv()

# Checkpoint contains LoRA adapters + heads + tokenizer.
# The base encoder weights come from HuggingFace (cached after first download).
BASE_MODEL    = os.getenv("BASE_MODEL", "answerdotai/ModernBERT-large")
CHECKPOINT    = os.getenv("MODEL_CHECKPOINT", "models/checkpoints/best_multitask")
TOX_THRESHOLD = float(os.getenv("TOX_THRESHOLD", "0.40"))   # tuned on val set (AUC=0.9176)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources on startup; clean up on shutdown."""
    print(f"Device     : {DEVICE}")
    print(f"Base model : {BASE_MODEL}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Tox thresh : {TOX_THRESHOLD}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Tokenizer is saved alongside the LoRA adapter in the checkpoint dir.
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # Build model shell with the base encoder (downloads ~400MB from HF on first run).
    # Then overlay the LoRA adapter and merge into the weights for clean inference.
    model = ContentModerationModel(
        encoder_name=BASE_MODEL,
        n_intents=len(INTENT_LABELS),
        focal_alpha=0.75,   # matches training config
        focal_gamma=1.0,
        alpha=0.65,
        beta=0.5,
        label_smoothing=0.1,
    )
    print("Merging LoRA adapter...")
    model.encoder = PeftModel.from_pretrained(model.encoder, CHECKPOINT).merge_and_unload()

    # Load classification heads (toxicity + intent linear layers)
    heads = torch.load(Path(CHECKPOINT) / "heads.pt", map_location=DEVICE)
    model.toxicity_head.load_state_dict(heads["toxicity_head"])
    model.intent_head.load_state_dict(heads["intent_head"])

    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    print("Model loaded.")

    # ── Trie (instant slur/keyword pre-filter) ─────────────────────────────────
    trie = SlurTrie()
    slur_file = Path("data/custom/slur_lexicon.txt")
    if slur_file.exists():
        trie.load_from_file(str(slur_file))

    # ── Redis (session intent history) ────────────────────────────────────────
    session_store = SessionStore()
    try:
        await session_store.connect()
        print("Redis connected.")
    except Exception as e:
        print(f"WARNING: Redis unavailable ({e}). Sessions won't persist across requests.")
        session_store = None

    # ── HMM (session risk scoring) ────────────────────────────────────────────
    hmm_path = Path("models/checkpoints/hmm.pkl")
    hmm = SessionHMM()
    if hmm_path.exists():
        hmm.load(hmm_path)
        print("HMM loaded.")
    else:
        print("WARNING: hmm.pkl not found. Run scripts/train_hmm.py first.")

    # ── PostgreSQL (message log + vector similarity) ───────────────────────────
    message_store = None
    try:
        message_store = MessageStore()
        await message_store.connect()
        await message_store.init_schema()
        print("PostgreSQL connected.")
    except Exception as e:
        print(f"WARNING: PostgreSQL unavailable ({e}). Messages won't be persisted.")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = ModerationPipeline(
        model=model,
        tokenizer=tokenizer,
        trie=trie,
        session_store=session_store or SessionStore(),  # in-memory fallback
        hmm=hmm,
        message_store=message_store,
        device=DEVICE,
        tox_threshold=TOX_THRESHOLD,
    )
    app.state.pipeline = pipeline
    app.state.session_store = session_store
    app.state.message_store = message_store

    print("Pipeline ready. Visit http://localhost:8000/docs")
    yield

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if session_store:
        await session_store.close()
    if message_store:
        await message_store.close()
    print("Shutdown complete.")


app = FastAPI(
    title="Content Moderation API",
    description="Real-time content moderation + intent classification",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")


# ── WebSocket: live classification feed ───────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active.copy():
            try:
                await ws.send_json(data)
            except Exception:
                self.active.remove(ws)


manager = ConnectionManager()
app.state.ws_manager = manager


@app.websocket("/ws/feed")
async def websocket_feed(ws: WebSocket):
    """Live moderation feed — dashboard subscribes here."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "checkpoint": CHECKPOINT,
        "tox_threshold": TOX_THRESHOLD,
    }
