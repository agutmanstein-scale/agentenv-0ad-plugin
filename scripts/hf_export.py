"""Builds the Hugging Face dataset's data files from finished matches, in the layout of the other AgentEnv game
datasets: the bundle's tasks, and one row per task, per match and per seat in parquet, plus each match's whole
results.json in raw/zeroad.jsonl. The dataset card (README.md) and the media are written by hand.

    uv run --with pyarrow python scripts/hf_export.py OUT MATCH_DIR_OR_RESULTS_JSON...

A match is named by its folder (or the file's stem), <task>-<yyyymmdd>-<hhmmss> as live/box/match.sh names it.
"""

import json
import re
import shutil
import sys
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

NAME = re.compile(r"^(?P<task>.+)-(?P<day>\d{8})-(?P<time>\d{6})$")
URL = re.compile(r"https?://[^\s'\"]+")


def load_match(path: Path) -> tuple[str, str, str, dict]:
    results = path / "results.json" if path.is_dir() else path
    name = path.name if path.is_dir() else path.stem
    m = NAME.match(name)
    if not m:
        sys.exit(f"{name}: not <task>-<yyyymmdd>-<hhmmss>")
    created = datetime.strptime(m["day"] + m["time"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return f"zeroad/{name}", m["task"], created.strftime("%Y-%m-%d %H:%M UTC"), json.loads(results.read_text())


def redacted(value):
    """The results with every URL as <url>: a failed model call's error names the model endpoint."""
    if isinstance(value, str):
        return URL.sub("<url>", value)
    if isinstance(value, list):
        return [redacted(v) for v in value]
    if isinstance(value, dict):
        return {k: redacted(v) for k, v in value.items()}
    return value


def of_type(steps: list[dict], kind: str) -> list[dict]:
    return [s for s in steps if s["type"] == kind]


def task_rows(tasks_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(tasks_dir.glob("*.json")):
        steps = json.loads(f.read_text())
        lobby = of_type(steps, "open_lobby")[0]["game_settings"]
        caps = sum(float(s.get("env_vars", {}).get("ZEROAD_MAX_COST_USD", 0)) for s in of_type(steps, "deploy_agent"))
        caps += sum(s["game_settings"].get("reflex_max_usd", 0) for s in of_type(steps, "add_player_slot"))
        prompts = of_type(steps, "prompt_agent")
        rows.append(
            {
                "task": f.stem,
                "task_id": f"zeroad/{f.stem}",
                "prompt": prompts[0]["prompt"] if prompts else None,
                "envs": sorted({s["env_id"] for s in steps if "env_id" in s}),
                "agents": sorted({s["a2a_agent_id"] for s in of_type(steps, "deploy_agent")}),
                "models": [s["model"] for s in prompts],
                "seats": len(of_type(steps, "add_player_slot")),
                "rendered": bool(lobby.get("render")),
                "time_cap_minutes": lobby["time_cap_minutes"],
                "speed": lobby["speed"],
                "caps_usd": round(caps, 2),
                "num_steps": len(steps),
                "steps": json.dumps(steps),
            }
        )
    return rows


def ranked(seats: list[dict]) -> dict[str, int]:
    order = sorted(seats, key=lambda s: (s["outcome"] != "won", s["state"] == "defeated", -s["score"]["total"]))
    return {s["name"]: i + 1 for i, s in enumerate(order)}


def episode_row(episode_id: str, task: str, created: str, r: dict) -> dict:
    seats = list(r["seats"].values())
    winners = [s["name"] for s in seats if s["outcome"] == "won"]
    return {
        "episode_id": episode_id,
        "task": task,
        "created_utc": created,
        "status": r["status"],
        "detail": r["detail"],
        "ended_by": r["ended_by"],
        "winner": winners[0] if len(winners) == 1 else None,
        "seats": len(seats),
        "models": [s.get("model") or s["kind"] for s in seats],
        "game_time_s": r["game_time_s"],
        "time_cap_s": r["time_cap_s"],
        "map": f"{r['map']['name']} {r['map']['size']} seed {r['map']['seed']}",
        "rendered": r["rendered"],
        "tick_errors": r["tick_errors"],
        "clock_overruns": r["clock"]["overruns"],
        "spend_usd_est": round(r["spend_usd_est"], 4),
        "scores": json.dumps({s["name"]: s["grade"] for s in seats}),
    }


def player_rows(episode_id: str, task: str, created: str, r: dict) -> list[dict]:
    seats = list(r["seats"].values())
    rank = ranked(seats)
    rows = []
    for s in seats:
        spend, metrics, combat = s.get("spend") or {}, s.get("metrics") or {}, s.get("combat") or {}
        latency = metrics.get("latency_s") or {}
        rows.append(
            {
                "episode_id": episode_id,
                "task": task,
                "created_utc": created,
                "player": s["player"],
                "name": s["name"],
                "kind": s["kind"],
                "civ": s["civ"],
                "model": s.get("model"),
                "reflex": "reflex" in s,
                "outcome": s["outcome"],
                "rank": rank[s["name"]],
                "ranked": len(seats),
                "reward": s["grade"],
                "score": s["score"]["total"],
                "score_economy": s["score"]["economy"],
                "score_military": s["score"]["military"],
                "score_exploration": s["score"]["exploration"],
                "decisions": s["moves"]["made"],
                "missed": s["moves"]["missed"],
                "latency_p50_s": latency.get("p50"),
                "latency_p95_s": latency.get("p95"),
                "tool_calls": (s.get("tool_calls") or {}).get("n"),
                "idle_villager_s": metrics.get("idle_villager_s"),
                "units_trained": combat.get("units_trained"),
                "units_lost": combat.get("units_lost"),
                "enemy_units_killed": combat.get("enemy_units_killed"),
                "buildings_lost": combat.get("buildings_lost"),
                "brain_usd_est": (spend.get("brain") or {}).get("usd_est"),
                "reflex_usd_est": (spend.get("reflex") or {}).get("usd_est"),
                "said": [line[1] if isinstance(line, list) else line for line in s.get("said") or []],
            }
        )
    return rows


def write(table: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(table), path)


def main(out: Path, matches: list[Path]) -> None:
    bundle = Path(str(files("agentenv_zeroad.bundles").joinpath("zeroad")))
    shutil.copytree(bundle, out / "bundles" / "zeroad", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__*"))
    write(task_rows(bundle / "tasks"), out / "tasks" / "zeroad.parquet")
    loaded = sorted((load_match(p) for p in matches), key=lambda m: m[2])
    write([episode_row(*m) for m in loaded], out / "episodes" / "zeroad.parquet")
    write([row for m in loaded for row in player_rows(*m)], out / "players" / "zeroad.parquet")
    raw = out / "raw" / "zeroad.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"episode_id": e, "task": t, "results": redacted(r)}) + "\n" for e, t, _, r in loaded)
    raw.write_text("".join(lines))
    print(f"{out}: {len(loaded)} matches")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
