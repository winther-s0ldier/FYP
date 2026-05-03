"""
Redis session store — sliding window of intent labels per user.
O(1) push + trim. TTL-based expiry (default 30 min inactivity).
"""
import json
import redis.asyncio as aioredis
import redis as syncredis
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 1800))
SESSION_WINDOW = int(os.getenv("SESSION_WINDOW_SIZE", 10))


class SessionStore:
    """Async Redis-backed session store."""

    def __init__(self, redis_url: str = REDIS_URL):
        self._url = redis_url
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = await aioredis.from_url(self._url, decode_responses=True)

    async def close(self):
        if self._client:
            await self._client.aclose()

    def _key_intents(self, user_id: str) -> str:
        return f"session:{user_id}:intents"

    def _key_risk(self, user_id: str) -> str:
        return f"session:{user_id}:risk"

    async def push_intent(
        self, user_id: str, intent_label: str, confidence: float,
        confidence_threshold: float = 0.70
    ):
        """Push intent label to session. Low-confidence predictions stored as 'uncertain'."""
        label = intent_label if confidence >= confidence_threshold else "uncertain"
        key = self._key_intents(user_id)
        pipe = self._client.pipeline()
        pipe.lpush(key, label)
        pipe.ltrim(key, 0, SESSION_WINDOW - 1)
        pipe.expire(key, SESSION_TTL)
        await pipe.execute()

    async def get_intents(self, user_id: str) -> list[str]:
        """Get last N intent labels for user (most recent first)."""
        return await self._client.lrange(self._key_intents(user_id), 0, -1) or []

    async def set_risk(self, user_id: str, risk_score: float):
        pipe = self._client.pipeline()
        pipe.set(self._key_risk(user_id), str(risk_score))
        pipe.expire(self._key_risk(user_id), SESSION_TTL)
        await pipe.execute()

    async def get_risk(self, user_id: str) -> float:
        val = await self._client.get(self._key_risk(user_id))
        return float(val) if val else 0.0

    async def clear_session(self, user_id: str):
        await self._client.delete(self._key_intents(user_id), self._key_risk(user_id))

    async def session_exists(self, user_id: str) -> bool:
        return bool(await self._client.exists(self._key_intents(user_id)))

    # --- Markov transition matrix (loaded at startup, updated nightly) ---

    async def load_transition_matrix(self, matrix: dict):
        """Store HMM transition matrix as Redis Hash."""
        flat = {f"{src}:{dst}": str(prob) for src, row in matrix.items()
                for dst, prob in row.items()}
        await self._client.hset("markov:transition_matrix", mapping=flat)

    async def get_transition_prob(self, from_intent: str, to_intent: str) -> float:
        val = await self._client.hget("markov:transition_matrix", f"{from_intent}:{to_intent}")
        return float(val) if val else 0.0


class SyncSessionStore:
    """Synchronous version for scripts and tests."""

    def __init__(self, redis_url: str = REDIS_URL):
        self._client = syncredis.from_url(redis_url, decode_responses=True)

    def push_intent(self, user_id: str, intent_label: str, confidence: float,
                    threshold: float = 0.70):
        label = intent_label if confidence >= threshold else "uncertain"
        key = f"session:{user_id}:intents"
        pipe = self._client.pipeline()
        pipe.lpush(key, label)
        pipe.ltrim(key, 0, SESSION_WINDOW - 1)
        pipe.expire(key, SESSION_TTL)
        pipe.execute()

    def get_intents(self, user_id: str) -> list[str]:
        return self._client.lrange(f"session:{user_id}:intents", 0, -1) or []

    def clear_session(self, user_id: str):
        self._client.delete(f"session:{user_id}:intents", f"session:{user_id}:risk")
