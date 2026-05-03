"""
LLM escalation router — Groq (primary) → GPT-4o-mini (fallback).
Called only for low-confidence messages (~5-10% of traffic).
"""
import json
import os
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a content moderation expert. Analyze messages for toxicity and intent.
Be precise, nuanced, and context-aware. Consider sarcasm, gaming slang, and cultural context.
Always return valid JSON only."""

USER_TEMPLATE = """Classify the following message for content moderation.

Message: "{message}"
Session history (last intents, most recent first): {session_history}
Similar past flagged cases: {similar_cases}
Relevant policy rules: {policy_context}

Return ONLY valid JSON:
{{
    "toxicity": <float 0.0-1.0>,
    "intent": "<intent_label>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<one sentence explanation>",
    "policy_rule": "<cited rule ID or null>"
}}

Intent labels available: {intent_labels}"""


async def escalate(
    message: str,
    session_history: list[str],
    similar_cases: list[dict],
    policy_context: str,
    intent_labels: list[str],
) -> dict:
    """Route to Groq (primary) or OpenAI (fallback). Returns parsed JSON dict."""
    prompt = USER_TEMPLATE.format(
        message=message,
        session_history=session_history[:5],
        similar_cases=similar_cases[:3],
        policy_context=policy_context[:500],
        intent_labels=", ".join(intent_labels),
    )

    # Try primary provider
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        result = await _call_groq(prompt)
        if result:
            return result

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        result = await _call_openai(prompt)
        if result:
            return result

    # Final fallback — return uncertain classification
    return {
        "toxicity": 0.5,
        "intent": "question",
        "confidence": 0.0,
        "reasoning": "LLM escalation failed — all providers unavailable",
        "policy_rule": None,
    }


async def _call_groq(prompt: str) -> Optional[dict]:
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=GROQ_API_KEY)
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512,
        )
        return _parse_response(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq escalation error: {e}")
        return None


async def _call_openai(prompt: str) -> Optional[dict]:
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512,
        )
        return _parse_response(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI escalation error: {e}")
        return None


def _parse_response(content: str) -> Optional[dict]:
    try:
        data = json.loads(content)
        # Validate required fields
        required = {"toxicity", "intent", "confidence", "reasoning"}
        if not required.issubset(data.keys()):
            return None
        data["toxicity"] = float(data["toxicity"])
        data["confidence"] = float(data["confidence"])
        return data
    except (json.JSONDecodeError, ValueError, KeyError):
        # Try extracting JSON from text if response has extra content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None
