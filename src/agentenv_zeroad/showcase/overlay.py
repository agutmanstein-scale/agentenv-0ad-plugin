"""The showcase's overlay: the env's overlay state plus the series header, the session table, the hold line, the
backstory card and the intermission card, served on loopback to the stream's own browser."""

import hmac
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from agentenv_zeroad.live import overlay_html, overlay_state
from agentenv_zeroad.server import LOOPBACK

POLICY_LABEL = "QWEN3-8B + 8 h RL"
BACKSTORY: tuple[str, ...] = (
    "RUN 1: 8 h on one box",
    "vs Petra  37.8% → 72.4% won",
    "vs Opus 13-3 · Astra 9-7",
    "43.7M of 8.2B weights trained (0.5%)",
    "50 updates · 1,462 games",
)


class SessionView(Protocol):
    """What the overlay reads of the session."""

    rows: Sequence[Any]  # each has `.result` ("win" | "loss" | "draw" | "no_result") and `.reason` (str | None)

    def table(self) -> list[dict]: ...


def hold_of(match: Any) -> dict | None:
    """The seat the lockstep clock is holding for: its model and how long the hold has lasted."""
    holding = getattr(getattr(match, "lockstep", None), "holding", None)
    if not holding:
        return None
    model = next((s["model"] for s in overlay_state(match)["seats"] if s["slot"] == holding["slot"]), None)
    return {"model": model, "waited_s": round(max(0.0, match.now() - holding["since"]), 1)}


def _latest_no_result_reason(session: SessionView) -> str | None:
    for row in reversed(list(session.rows)):
        if row.result == "no_result":
            return row.reason
    return None


def _last_game(last: dict) -> dict:
    """The last game's card line. `result` is win (default), draw or no_result; a win names `winner`, `loser`, `how`."""
    result = last.get("result", "win")
    if result == "draw":
        headline = "Draw · net worth tied at 30:00"
    elif result == "no_result":
        headline = f"No result · {last.get('reason') or 'the game could not be scored'}"
    else:
        verb = f"{last['winner']} beat {last['loser']}" if last.get("loser") else f"{last['winner']} won"
        headline = f"{verb} · {last['how']}"
    return {**last, "headline": headline}


def showcase_state(
    session: SessionView,
    match: Any,
    *,
    phase: str,
    opponent: str,
    game_no: int,
    round_no: int,
    hold: dict | None,
    mode: str,
    played_at: str | None,
    intermission: dict | None,
    dropped: Sequence[str] = (),
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`overlay_state(match)` (or `base`, a replay's game block, or nothing) plus the showcase's blocks. Never holds a
    secret."""
    state: dict[str, Any] = overlay_state(match) if match is not None else dict(base or {})
    table = session.table()
    state["phase"] = phase
    state["series"] = {
        "policy": POLICY_LABEL,
        "opponent": opponent,
        "game": game_no,
        "round": round_no,
        "clock": state.get("clock", "00:00"),
    }
    notes = [f"{name} skipped for the rest of the session (2 no-results in a row)" for name in dropped]
    state["session"] = {"table": table, "no_result_reason": _latest_no_result_reason(session), "dropped": notes}
    state["backstory"] = BACKSTORY
    state["hold"] = hold
    state["intermission"] = (
        {**intermission, "last": _last_game(intermission["last"]), "table": intermission.get("table", table)}
        if intermission
        else None
    )
    label = (played_at or "").removesuffix(" ET")
    state["replay"] = {"played_at": f"{label} ET"} if mode == "replay" and label else None
    return state


class OverlayServer:
    """The overlay page and its state on loopback, for the stream's browser alone."""

    def __init__(
        self, state: Callable[[], dict[str, Any]], token: str, host: str = "127.0.0.1", port: int = 18766
    ) -> None:
        self._state = state
        self._token = token
        self._server = uvicorn.Server(uvicorn.Config(self.app(), host=host, port=port, log_level="warning"))
        self._thread: threading.Thread | None = None

    def app(self) -> Starlette:
        return Starlette(routes=[Route("/live", self._page), Route("/live/state.json", self._state_json)])

    async def _page(self, request: Request) -> HTMLResponse:
        return HTMLResponse(overlay_html())

    async def _state_json(self, request: Request) -> JSONResponse:
        host = request.client.host if request.client is not None else ""
        token = request.query_params.get("token", "")
        if host not in LOOPBACK or not hmac.compare_digest(token.encode(), self._token.encode()):
            message = "the overlay's data goes only to the stream's own browser"
            return JSONResponse({"ok": False, "error": {"code": "forbidden", "message": message}}, status_code=403)
        return JSONResponse(self._state(), headers={"Cache-Control": "no-store"})

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="showcase-overlay")
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._server.started:
            raise RuntimeError("the overlay server did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
