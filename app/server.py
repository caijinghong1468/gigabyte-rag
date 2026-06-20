"""FastAPI server with streaming (SSE) answers — the Docker entry point.

Run locally:   uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
Health check:  GET  /healthz
Ask (stream):  GET  /chat?question=螢幕更新率是多少
               POST /chat   {"question": "...", "top_k": 5}

Each stream emits `data: {"token": "..."}` events, then a final
`data: {"done": true, "metrics": {...}, "sources": [...]}` event.
"""

import json
import pathlib
import sys
import threading
from collections import OrderedDict, deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from rag.config import cfg  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

app = FastAPI(title="AORUS MASTER 16 AM6H Spec RAG")
FRONTEND = pathlib.Path(__file__).resolve().parent / "frontend.html"

_pipe: RagPipeline | None = None
_lock = threading.Lock()


def get_pipe() -> RagPipeline:
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                _pipe = RagPipeline.from_artifacts()
    return _pipe


# --- Conversation memory --------------------------------------------------- #
# Per-session rolling history, keyed by a session_id the browser generates. Each
# session keeps the last MAX_TURNS messages (the pipeline further trims this to
# the token budget per request). MAX_SESSIONS bounds total memory: oldest
# session is evicted first (this is a single-box demo, not a multi-tenant store).
# 16 messages = 8 exchanges: with N_CTX=8192 the token budget can actually hold
# this many, so the cap and the budget are matched (a smaller cap would waste
# the larger window; the budget still trims if answers run long).
MAX_TURNS = 16         # 8 user/assistant exchanges
MAX_SESSIONS = 200
_sessions: "OrderedDict[str, deque]" = OrderedDict()


def _history(session_id: str) -> deque:
    hist = _sessions.get(session_id)
    if hist is None:
        hist = deque(maxlen=MAX_TURNS)
        _sessions[session_id] = hist
        if len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)  # evict least-recently-created
    _sessions.move_to_end(session_id)
    return hist


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None
    session_id: str | None = None


def _sse(pipe: RagPipeline, question: str, top_k: int | None, session_id: str | None):
    # Snapshot prior turns BEFORE generating (we append this turn afterwards).
    hist = _history(session_id) if session_id else None
    prior = list(hist) if hist is not None else None

    for token in pipe.answer_stream(question, top_k=top_k, history=prior):
        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
    last = pipe.last

    # Persist this exchange so the next question in the session can see it.
    if hist is not None:
        hist.append({"role": "user", "content": question})
        hist.append({"role": "assistant", "content": last["answer"]})

    final = {
        "done": True,
        "answer": last["answer"],
        "metrics": last["metrics"],
        "retrieval_s": last["retrieval_s"],
        "sources": [
            {"id": r["chunk"]["id"], "category": r["chunk"]["category"], "score": round(r["score"], 4)}
            for r in last["retrieved"]
        ],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": str(cfg.model_path.name), "thinking": cfg.enable_thinking}


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.post("/chat")
def chat_post(req: ChatRequest):
    pipe = get_pipe()
    return StreamingResponse(
        _sse(pipe, req.question, req.top_k, req.session_id), media_type="text/event-stream"
    )


@app.get("/chat")
def chat_get(question: str, top_k: int | None = None, session_id: str | None = None):
    pipe = get_pipe()
    return StreamingResponse(
        _sse(pipe, question, top_k, session_id), media_type="text/event-stream"
    )


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    """Forget a conversation's history (the browser's "新對話" button)."""
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
