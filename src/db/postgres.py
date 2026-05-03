"""
PostgreSQL + pgvector — message storage and similarity search.
Schema mirrors FYP.md §7.2.
"""
import asyncpg
import numpy as np
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:secret@localhost:5432/moderation")

CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    toxicity_score  FLOAT NOT NULL,
    intent_label    TEXT NOT NULL,
    confidence      FLOAT NOT NULL,
    session_risk    FLOAT NOT NULL DEFAULT 0.0,
    action          TEXT NOT NULL CHECK (action IN ('allow','warn','flag','block')),
    api_escalated   BOOLEAN DEFAULT FALSE,
    trie_matched    BOOLEAN DEFAULT FALSE,
    latency_ms      FLOAT,
    reasoning       TEXT,
    policy_rule     TEXT,
    embedding       vector(768)
);

CREATE INDEX IF NOT EXISTS messages_hnsw_idx
    ON messages USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS messages_user_time_idx
    ON messages (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS messages_action_time_idx
    ON messages (action, timestamp DESC);

CREATE TABLE IF NOT EXISTS intents (
    id          SERIAL PRIMARY KEY,
    label       TEXT UNIQUE NOT NULL,
    category    TEXT NOT NULL CHECK (category IN ('benign', 'suspicious', 'malicious')),
    risk_weight FLOAT NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS mod_actions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    moderator  TEXT,
    action     TEXT,
    notes      TEXT,
    timestamp  TIMESTAMPTZ DEFAULT NOW()
);
"""

SEED_INTENTS_SQL = """
INSERT INTO intents (label, category, risk_weight) VALUES
    ('greeting', 'benign', 0.0), ('question', 'benign', 0.0),
    ('information_request', 'benign', 0.0), ('small_talk', 'benign', 0.0),
    ('feedback', 'benign', 0.0), ('help_request', 'benign', 0.0),
    ('joke', 'benign', 0.0), ('farewell', 'benign', 0.0),
    ('personal_probe', 'suspicious', 0.4), ('repeated_contact', 'suspicious', 0.5),
    ('boundary_testing', 'suspicious', 0.5), ('unusual_urgency', 'suspicious', 0.4),
    ('threat', 'malicious', 0.95), ('harassment', 'malicious', 0.80),
    ('hate_speech', 'malicious', 0.85), ('grooming_signal', 'malicious', 0.90),
    ('doxxing_attempt', 'malicious', 0.95), ('coordinated_abuse', 'malicious', 0.90)
ON CONFLICT (label) DO NOTHING;
"""


class MessageStore:
    def __init__(self, dsn: str = DATABASE_URL):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self, min_size: int = 5, max_size: int = 20):
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=min_size, max_size=max_size
        )

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def init_schema(self):
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
            await conn.execute(SEED_INTENTS_SQL)
        print("Database schema initialised.")

    async def save(self, response) -> str:
        """Persist a ClassificationResponse. Returns the message UUID."""
        embedding_str = (
            "[" + ",".join(f"{v:.6f}" for v in response.embedding) + "]"
            if response.embedding else None
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (
                    user_id, session_id, text, toxicity_score, intent_label,
                    confidence, session_risk, action, api_escalated, trie_matched,
                    latency_ms, reasoning, policy_rule, embedding
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::vector)
                RETURNING id
                """,
                response.user_id, getattr(response, "session_id", ""),
                response.text, response.toxicity_score, response.intent_label,
                response.confidence, response.session_risk, response.action,
                response.api_escalated, response.trie_matched,
                response.latency_ms, response.reasoning, response.policy_rule,
                embedding_str,
            )
        return str(row["id"])

    async def count_similar_flagged(
        self, embedding: np.ndarray, threshold: float = 0.85, limit: int = 5
    ) -> int:
        """Count past flagged/blocked messages with cosine similarity above threshold."""
        embedding_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id FROM messages
                WHERE action IN ('flag', 'block')
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::vector) > $2
                LIMIT $3
                """,
                embedding_str, threshold, limit,
            )
        return len(rows)

    async def get_recent_flagged(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT text, toxicity_score, intent_label, action, timestamp
                FROM messages
                WHERE user_id = $1 AND action IN ('flag', 'block')
                ORDER BY timestamp DESC LIMIT $2
                """,
                user_id, limit,
            )
        return [dict(r) for r in rows]
