"""
LLM Analyst Chat — Gemini-powered explanation endpoint
--------------------------------------------------------
Lets a user ask free-form questions about a scored transaction
(or general fraud patterns) and get an answer grounded in the
model's actual SHAP explanation + risk score, not a generic guess.
"""

import os
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["AI Analyst Chat"])

# Preferred models in order of priority (starting with the currently recommended 3.6-flash)
CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro"
]


class ChatRequest(BaseModel):
    question: str
    transaction: Optional[Dict[str, Any]] = None
    prediction_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    answer: str
    model_used: str


def get_configured_gemini_client():
    """Dynamically configure and return the API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return api_key


def build_prompt(question: str, prediction_context: dict) -> str:
    """
    Ground the LLM strictly in the model's own output so it explains
    the *actual* prediction instead of hallucinating a generic answer.
    """
    risk_score = prediction_context.get("risk_score_pct", "unknown")
    risk_tier = prediction_context.get("risk_tier", "unknown")
    decision = prediction_context.get("decision", "unknown")
    explanation_points = prediction_context.get("explanation", [])
    explanation_text = (
        "\n".join(f"- {e}" for e in explanation_points)
        if explanation_points
        else "No specific risk factors were flagged."
    )

    prompt = f"""You are a fraud risk analyst assistant embedded in a payment fraud detection dashboard.
A machine learning model (XGBoost) has already scored a transaction. Your job is to answer the
analyst's question using ONLY the information below. Do not invent numbers, policies, or facts
that aren't given here. If the question can't be answered from this data, say so plainly.

MODEL OUTPUT FOR THIS TRANSACTION:
- Risk score: {risk_score}%
- Risk tier: {risk_tier}
- Recommended decision: {decision}
- Key risk factors identified by the model (via SHAP):
{explanation_text}

ANALYST QUESTION:
{question}

Answer in 2-4 concise sentences. Be direct and practical, like a colleague explaining a flag,
not a generic chatbot. If the risk factors don't clearly answer the question, say what additional
information would be needed."""
    return prompt


def generate_with_gemini(prompt: str) -> tuple[str, str]:
    """
    Iterate through candidate models and dynamically discover supported models
    to guarantee high availability across different Gemini API versions.
    """
    last_error = None

    # 1. Try candidate list first (gemini-3.6-flash, etc.)
    for m_name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip(), m_name
        except Exception as e:
            last_error = e
            continue

    # 2. Dynamic discovery fallback: query active models supporting generateContent on this key
    try:
        for m in genai.list_models():
            if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                model_name_clean = m.name.replace("models/", "")
                try:
                    model = genai.GenerativeModel(model_name_clean)
                    response = model.generate_content(prompt)
                    if response and response.text:
                        return response.text.strip(), model_name_clean
                except Exception as e:
                    last_error = e
                    continue
    except Exception as e:
        last_error = e

    raise HTTPException(
        status_code=502,
        detail=f"Gemini API error: {str(last_error)}"
    )


@router.post("/chat", response_model=ChatResponse)
def chat_with_analyst(req: ChatRequest):
    api_key = get_configured_gemini_client()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is not configured on the server. "
                "Please add GEMINI_API_KEY to your environment variables or Render dashboard."
            )
        )

    if not req.prediction_context and not req.transaction:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'prediction_context' (from a prior /predict call) or 'transaction' fields."
        )

    prediction_context = req.prediction_context
    if prediction_context is None:
        raise HTTPException(
            status_code=400,
            detail="Pass 'prediction_context' from your /predict response."
        )

    prompt = build_prompt(req.question, prediction_context)
    answer, model_used = generate_with_gemini(prompt)

    return ChatResponse(answer=answer, model_used=model_used)
