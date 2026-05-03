"""
POST /api/v1/classify — main classification endpoint.
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.pipeline import ClassificationRequest

router = APIRouter(tags=["classification"])


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(default="")


class ClassifyResponse(BaseModel):
    action: str
    toxicity_score: float
    intent_label: str
    confidence: float
    session_risk: float
    risk_score: float
    trie_matched: bool
    api_escalated: bool
    latency_ms: float
    reasoning: str | None = None
    policy_rule: str | None = None


@router.post("/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest, request: Request):
    pipeline = request.app.state.pipeline
    ws_manager = getattr(request.app.state, "ws_manager", None)

    result = await pipeline.classify(
        ClassificationRequest(
            text=body.text,
            user_id=body.user_id,
            session_id=body.session_id,
        )
    )

    response = ClassifyResponse(
        action=result.action,
        toxicity_score=round(result.toxicity_score, 4),
        intent_label=result.intent_label,
        confidence=round(result.confidence, 4),
        session_risk=round(result.session_risk, 4),
        risk_score=round(result.risk_score, 4),
        trie_matched=result.trie_matched,
        api_escalated=result.api_escalated,
        latency_ms=round(result.latency_ms, 2),
        reasoning=result.reasoning,
        policy_rule=result.policy_rule,
    )

    # Push to WebSocket feed for live dashboard
    if ws_manager:
        await ws_manager.broadcast({
            "user_id": body.user_id,
            **response.model_dump(),
        })

    return response
