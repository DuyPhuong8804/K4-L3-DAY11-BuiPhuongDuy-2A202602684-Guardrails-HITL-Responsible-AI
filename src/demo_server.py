"""
Lab 11 — Live demo server for the Blue guardrail pipeline.

Runs the REAL pipeline built in Checkpoint 2/3 (rate limiter, input
guardrail, output guardrail, audit log, monitoring) behind a small chat
UI so you can see the guardrails work interactively instead of only via
``outputs/results.json``.

Run from repo root:

    uvicorn src.demo_server:app --reload --port 8000

Then open http://127.0.0.1:8000 in a browser. Requires the same
``.env`` (OPENROUTER_API_KEY) used by ``python src/main.py``.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google.genai import types

from agents.agent import create_blue_agent
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import build_production_plugins, _content_to_text
from core.utils import chat_with_agent


app = FastAPI(title="VinBank Blue Guardrail Demo")

_STATE: dict = {}


@app.on_event("startup")
async def _startup():
    plugins = build_production_plugins()
    rate_limiter, input_guardrail, output_guardrail = plugins
    audit = AuditLogPlugin()
    monitor = MonitoringAlert()
    agent, runner = create_blue_agent([])
    _STATE.update(
        {
            "rate_limiter": rate_limiter,
            "input_guardrail": input_guardrail,
            "output_guardrail": output_guardrail,
            "audit": audit,
            "monitor": monitor,
            "agent": agent,
            "runner": runner,
            "request_counter": 0,
            "recent": [],  # small in-memory feed for the UI side panel
        }
    )


class ChatIn(BaseModel):
    user_id: str
    message: str


class UserContext:
    def __init__(self, user_id: str):
        self.user_id = user_id


@app.post("/api/chat")
async def chat(payload: ChatIn):
    text = (payload.message or "").strip()
    user_id = (payload.user_id or "anon").strip() or "anon"
    if not text:
        return JSONResponse({"error": "empty message"}, status_code=400)

    rate_limiter = _STATE["rate_limiter"]
    input_guardrail = _STATE["input_guardrail"]
    output_guardrail = _STATE["output_guardrail"]
    audit = _STATE["audit"]
    monitor = _STATE["monitor"]
    agent = _STATE["agent"]
    runner = _STATE["runner"]

    _STATE["request_counter"] += 1
    request_id = f"demo-{_STATE['request_counter']:04d}"
    started = time.time()

    audit.record_input(user_id=user_id, text=text, request_id=request_id)
    monitor.total_requests += 1

    context = UserContext(user_id)
    content = types.Content(role="user", parts=[types.Part.from_text(text=text)])

    blocked_layer = None
    blocked_response = None
    for layer_name, plugin in (
        ("rate_limiter", rate_limiter),
        ("input_guardrail", input_guardrail),
    ):
        blocked_response = await plugin.on_user_message_callback(
            invocation_context=context, user_message=content
        )
        if blocked_response is not None:
            blocked_layer = layer_name
            break

    issues: list[str] = []
    if blocked_response is not None:
        reply_text = _content_to_text(blocked_response)
        blocked = True
        if blocked_layer == "rate_limiter":
            monitor.rate_limit_hits += 1
        monitor.blocked_requests += 1
    else:
        try:
            raw_text, _ = await chat_with_agent(agent, runner, text)
        except Exception as exc:  # OpenRouter/OpenAI errors (rate limit, timeout, ...)
            latency_ms = round((time.time() - started) * 1000)
            audit.record_output(
                user_id=user_id, text=str(exc), blocked=True,
                layer="llm_error", request_id=request_id,
            )
            entry = {
                "request_id": request_id,
                "user_id": user_id,
                "input": text,
                "reply": f"Lỗi khi gọi LLM (Blue/OpenRouter): {exc}",
                "blocked": True,
                "layer": "llm_error",
                "issues": [],
                "latency_ms": latency_ms,
            }
            _STATE["recent"].insert(0, entry)
            return JSONResponse(entry, status_code=200)
        response = type(
            "R", (), {"content": types.Content(
                role="model", parts=[types.Part.from_text(text=raw_text or "")]
            )}
        )()
        redacted_before = getattr(output_guardrail, "redacted_count", 0)
        filtered = await output_guardrail.after_model_callback(
            callback_context=context, llm_response=response
        )
        reply_text = _content_to_text(filtered.content)
        blocked = False
        blocked_layer = None
        if getattr(output_guardrail, "redacted_count", 0) > redacted_before:
            issues.append("redacted PII/secret")

    latency_ms = round((time.time() - started) * 1000)
    audit.record_output(
        user_id=user_id,
        text=reply_text,
        blocked=blocked,
        layer=blocked_layer,
        request_id=request_id,
    )

    entry = {
        "request_id": request_id,
        "user_id": user_id,
        "input": text,
        "reply": reply_text,
        "blocked": blocked,
        "layer": blocked_layer,
        "issues": issues,
        "latency_ms": latency_ms,
    }
    _STATE["recent"].insert(0, entry)
    _STATE["recent"] = _STATE["recent"][:50]

    return entry


@app.get("/api/stats")
async def stats():
    monitor: MonitoringAlert = _STATE["monitor"]
    alerts = monitor.check_metrics() or []
    return {
        "total_requests": monitor.total_requests,
        "blocked_requests": monitor.blocked_requests,
        "rate_limit_hits": monitor.rate_limit_hits,
        "alerts": [str(a) for a in alerts],
        "recent": _STATE["recent"][:10],
    }


_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "chatbox.html"))
