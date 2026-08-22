"""
service.py

HTTP wrapper around FoggyEngine.handle_message() for node deployment.

⚠️ PLACEHOLDER CONTRACT — the route path, request schema, and response
schema below are reasonable defaults, NOT confirmed against the real
Go backend / Temboa contract doc. Update this file once that doc's
§04 (endpoint, auth, request/response JSON shapes) is available — treat
everything in this file as "our best guess, ready to be corrected,"
not as final.

This process does NOT load the LLM itself — it calls llama-server (see
LLAMA_SERVER_URL in inference_pipeline_qwen_vlm.py, defaults to
127.0.0.1:8080) which Temboa's node agent is responsible for running,
per the "node agent owns model weights + llama.cpp, we own orchestration"
split. This service must be able to reach that address at runtime — same
Docker network / host networking, whichever the node agent expects.
"""

import os
import sys
import base64
import logging
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_pipeline_qwen_vlm import FoggyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoggyService")

engine: Optional[FoggyEngine] = None


class UTF8JSONResponse(JSONResponse):
    """Starlette's default JSONResponse sends `Content-Type: application/json`
    with no explicit charset -- valid per the JSON spec (RFC 8259 mandates
    UTF-8 by default), but PowerShell 5.1's Invoke-RestMethod does not
    reliably assume UTF-8 for a charset-less application/json response,
    confirmed as the actual cause of mojibake (em dashes, degree signs)
    reaching farmer-facing responses: the underlying Python string was
    verified correct via a repr() dump immediately after parsing
    llama-server's reply, ruling out everything upstream of this response
    being sent. Declaring the charset explicitly removes the ambiguity for
    any client, not just the one this was caught with."""
    media_type = "application/json; charset=utf-8"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Loading FoggyEngine (SigLIP2 + retriever + Whisper)...")
    engine = FoggyEngine()
    logger.info("FoggyEngine ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Foggy BSF Assistant", lifespan=lifespan, default_response_class=UTF8JSONResponse)


# --- Placeholder request/response schema — replace once the real contract is confirmed ---

class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    history: Optional[list[ChatTurn]] = None


class ChatResponse(BaseModel):
    response: str
    history: list[ChatTurn]


@app.get("/health")
def health():
    """Basic liveness check — does NOT confirm llama-server is reachable,
    only that this process started. Consider hitting FoggyEngine's own
    llama-server health check result here once there's a real need for
    deeper readiness checks."""
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not yet loaded.")

    image_path = None
    audio_path = None

    try:
        if req.image_base64:
            image_bytes = base64.b64decode(req.image_base64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(image_bytes)
            tmp.close()
            image_path = tmp.name

        if req.audio_base64:
            audio_bytes = base64.b64decode(req.audio_base64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
            tmp.write(audio_bytes)
            tmp.close()
            audio_path = tmp.name

        history_list = [t.model_dump() for t in req.history] if req.history else None

        response_text, updated_history = engine.handle_message(
            text=req.text,
            image_path=image_path,
            audio_path=audio_path,
            history=history_list,
        )

        return ChatResponse(
            response=response_text,
            history=[ChatTurn(**h) for h in updated_history],
        )

    finally:
        # Deliberately deleted here, not left for the caller — per the
        # earlier API team conversation, media deletion after processing
        # is expected behavior; doing it here means it happens regardless
        # of which team ends up owning what upstream of this service.
        for path in (image_path, audio_path):
            if path and os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))