"""lobot web server: FastAPI over the engine, serves the web/ frontend.

Same before/after loop as app.py (Gradio), with a custom frontend that
follows docs/DESIGN.md. Engine state (warm model, presets, active model) stays
in app.py so both UIs load and steer the same way.

Endpoints and shapes: docs/reference/api.md.

Shared state lives in server_state.STATE. Request shapes live in
server_schemas. Helpers and stream builders live in server_helpers.
Routes live in routers/, one module per area.

Run:
    uv run python -m model_lobotomy.server                            # loads weights, serves localhost:7860
    MODEL_LOBOTOMY_NO_MODEL=1 uv run python -m model_lobotomy.server  # UI only; runs return 503
"""

from __future__ import annotations

if __name__ == "__main__":
    # `python -m model_lobotomy.server`: read .env before any setting is read.
    from model_lobotomy.config import load_env_file
    _ENV_SET = load_env_file()

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from model_lobotomy import app as engine
from model_lobotomy import server_helpers as helpers
from model_lobotomy.llama_backend import UnsupportedError
from model_lobotomy.routers import (
    chat as _r_chat,
    create as _r_create,
    drafts_agent as _r_drafts,
    evidence as _r_evidence,
    install_verify as _r_install,
    reader_features as _r_reader,
    run as _r_run,
    tune as _r_tune,
)
from model_lobotomy.server_state import STATE

api = FastAPI(title="lobot", docs_url=None, redoc_url=None)
api.state.app_state = STATE


@api.exception_handler(UnsupportedError)
async def _unsupported_handler(_req, exc: UnsupportedError):
    """Backstop: a missed torch-only call is a 501, never a 500.

    Torch never raises UnsupportedError, so torch behavior does not
    change. Explicit _need_torch gates stay mandatory: this cannot
    help after a stream has started.
    """
    return JSONResponse(status_code=501, content={"detail": str(exc)})


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _hostname(host: str) -> str:
    """'127.0.0.1:7860' -> '127.0.0.1'; '[::1]:7860' -> '::1'."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:host.find("]")]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class SameOriginOnly:
    """Reject cross-origin POSTs from browsers (D5), DNS rebinding included.

    A browser POST carries Origin. It passes only when Origin equals the
    server's own scheme://host and that host is a loopback name: a rebound
    domain (evil.example resolving to 127.0.0.1) matches itself, so the
    loopback check is what stops it. Requests without Origin (curl, tests)
    pass. Plain ASGI, so streamed runs and Stop pass through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            origin = headers.get("origin")
            if origin is not None:
                host = headers.get("host", "")
                own = f"{scope.get('scheme', 'http')}://{host}".lower().rstrip("/")
                if origin.strip().lower().rstrip("/") != own or _hostname(host) not in LOOPBACK_HOSTS:
                    response = JSONResponse({"detail": "Cross-origin POST rejected."}, status_code=403)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


api.add_middleware(SameOriginOnly)

for _r in (_r_run, _r_create, _r_evidence, _r_drafts, _r_chat,
           _r_install, _r_reader, _r_tune):
    api.include_router(_r.router)

# Last, so /api/* routes win. web/ at the root keeps the page's relative
# paths (vendor/, example.json) the same under any static server.
api.mount("/", StaticFiles(directory=STATE.web, html=True), name="web")


if __name__ == "__main__":
    if STATE.replay:
        print(f"[server] REPLAY set: no weights; {len(helpers._recordings())} recorded runs "
              "answer /api/run", flush=True)
    elif STATE.no_model:
        print("[server] NO_MODEL set: serving the UI without weights", flush=True)
    else:
        engine.load_engine()
        print(f"[server] engine warm ({engine._ENGINE.model_id}); models: "
              f"{sorted(engine._PRESETS)}", flush=True)
    m = helpers._meta_api()
    print(f"[server] Meta API: {'on, ' + m['model'] if m['key'] else 'off (set META_API_KEY in .env)'}"
          f"{' (from .env)' if 'META_API_KEY' in _ENV_SET else ''}", flush=True)
    from model_lobotomy.config import get_settings
    uvicorn.run(api, host="127.0.0.1", port=get_settings().port)
