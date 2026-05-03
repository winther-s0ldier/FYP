"""
FastAPI application — REST + WebSocket endpoints.
Mount: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
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

CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "models/checkpoints/best_multitask")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources on startup; clean up on shutdown."""
    print(f"Starting on device: {DEVICE}")

    # --- Model ---
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = ContentModerationModel(
        encoder_name=CHECKPOINT, n_intents=len(INTENT_LABELS)
    )
    heads = torch.load(Path(CHECKPOINT) / "heads.pt", map_location=DEVICE)
    model.toxicity_head.load_state_dict(heads["toxicity_head"])
    model.intent_head.load_state_dict(heads["intent_head"])
    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    print("Model loaded.")

    # --- Trie ---
    trie = SlurTrie()
    slur_file = Path("data/custom/slur_lexicon.txt")
    if slur_file.exists():
        trie.load_from_file(str(slur_file))

    # --- Redis ---
    session_store = SessionStore()
    await session_store.connect()
    print("Redis connected.")

    # --- HMM ---
    hmm_path = Path("models/checkpoints/hmm.pkl")
    hmm = SessionHMM()
    if hmm_path.exists():
        hmm.load(hmm_path)
        print("HMM loaded.")
    else:
        print("WARNING: HMM model not found. Session risk will be 0. Run scripts/train_hmm.py")

    # --- PostgreSQL ---
    message_store = MessageStore()
    await message_store.connect()
    await message_store.init_schema()
    print("PostgreSQL connected.")

    # --- Pipeline ---
    pipeline = ModerationPipeline(
        model=model, tokenizer=tokenizer, trie=trie,
        session_store=session_store, hmm=hmm,
        message_store=message_store, device=DEVICE,
    )
    app.state.pipeline = pipeline
    app.state.session_store = session_store
    app.state.message_store = message_store

    print("Pipeline ready.")
    yield

    # --- Cleanup ---
    await session_store.close()
    await message_store.close()
    print("Shutdown complete.")


app = FastAPI(
    title="Content Moderation API",
    description="Real-time content moderation + intent classification",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")


# --- WebSocket: live classification feed ---
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
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
            # Keep connection alive; data is pushed via broadcast()
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}
