"""Local web app: `trader app` serves the dashboard at http://127.0.0.1:8000.

No login: keep it on your own machine, or reach it from your phone over a private network
such as Tailscale (`trader app --host 0.0.0.0`). Never expose it to the open internet.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import dashboard

WEB = Path(__file__).parent / "web"


def create(cfg: dict, demo: bool = False) -> FastAPI:
    market = dashboard.DemoMarket(cfg) if demo else dashboard.Market(cfg)
    lock = threading.Lock()  # one calculation at a time; pandas work is not worth parallelising here
    app = FastAPI(title="trader", docs_url=None, redoc_url=None)

    def answer(fn, *args):
        with lock:
            try:
                return JSONResponse(dashboard.jsonable(fn(*args)))
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/overview")
    def overview():
        return answer(dashboard.overview, market)

    @app.get("/api/chart/{name}")
    def chart(name: str, tf: str = "1d"):
        if tf not in ("1d", "1h"):
            raise HTTPException(400, "tf must be 1d or 1h")
        return answer(dashboard.chart, market, name, tf)

    @app.get("/api/check/{name}")
    def check(name: str, direction: str = "long"):
        if direction not in ("long", "short"):
            raise HTTPException(400, "direction must be long or short")
        return answer(dashboard.check, market, name, 1 if direction == "long" else -1)

    @app.post("/api/refresh")
    def refresh():
        with lock:
            market.refresh()
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def serve(cfg: dict, host: str = "127.0.0.1", port: int = 8000, demo: bool = False, browser: bool = True) -> None:
    import uvicorn

    if browser:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, [f"http://127.0.0.1:{port}"]).start()
    uvicorn.run(create(cfg, demo), host=host, port=port, log_level="warning")
