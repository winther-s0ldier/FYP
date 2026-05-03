"""
GET /api/v1/dashboard/* — analytics endpoints for the React dashboard.
"""
from fastapi import APIRouter, Request, Query
from datetime import datetime, timedelta

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/stats")
async def get_stats(request: Request, hours: int = Query(default=1, ge=1, le=168)):
    """Aggregate stats for the last N hours."""
    store = request.app.state.message_store
    since = datetime.utcnow() - timedelta(hours=hours)

    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                COUNT(*) FILTER (WHERE action = 'allow')  AS allowed,
                COUNT(*) FILTER (WHERE action = 'warn')   AS warned,
                COUNT(*) FILTER (WHERE action = 'flag')   AS flagged,
                COUNT(*) FILTER (WHERE action = 'block')  AS blocked,
                COUNT(*) FILTER (WHERE api_escalated)     AS escalated,
                AVG(latency_ms)                           AS avg_latency_ms,
                AVG(toxicity_score)                       AS avg_toxicity,
                COUNT(*)                                  AS total
            FROM messages
            WHERE timestamp > $1
            """,
            since,
        )
    row = dict(rows[0]) if rows else {}
    return {
        "period_hours": hours,
        "total": row.get("total", 0),
        "allowed": row.get("allowed", 0),
        "warned": row.get("warned", 0),
        "flagged": row.get("flagged", 0),
        "blocked": row.get("blocked", 0),
        "escalated": row.get("escalated", 0),
        "avg_latency_ms": round(row.get("avg_latency_ms") or 0, 2),
        "avg_toxicity": round(row.get("avg_toxicity") or 0, 4),
    }


@router.get("/dashboard/intent-distribution")
async def get_intent_distribution(
    request: Request, hours: int = Query(default=1, ge=1, le=168)
):
    store = request.app.state.message_store
    since = datetime.utcnow() - timedelta(hours=hours)

    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT intent_label, COUNT(*) as count
            FROM messages WHERE timestamp > $1
            GROUP BY intent_label ORDER BY count DESC
            """,
            since,
        )
    return [{"intent": r["intent_label"], "count": r["count"]} for r in rows]


@router.get("/dashboard/recent-flagged")
async def get_recent_flagged(
    request: Request, limit: int = Query(default=20, ge=1, le=100)
):
    store = request.app.state.message_store
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT user_id, text, toxicity_score, intent_label, session_risk,
                   action, api_escalated, reasoning, timestamp
            FROM messages
            WHERE action IN ('flag', 'block')
            ORDER BY timestamp DESC LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


@router.get("/dashboard/user-risk/{user_id}")
async def get_user_risk_timeline(
    user_id: str, request: Request,
    limit: int = Query(default=50, ge=1, le=200)
):
    store = request.app.state.message_store
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT timestamp, toxicity_score, session_risk, intent_label, action
            FROM messages
            WHERE user_id = $1
            ORDER BY timestamp DESC LIMIT $2
            """,
            user_id, limit,
        )
    return [dict(r) for r in rows]
