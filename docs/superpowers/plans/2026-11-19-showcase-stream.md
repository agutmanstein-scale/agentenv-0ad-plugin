# Showcase Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 1's final adapter (v50) plays Opus, Astra and Petra-3 one game at a time, rendered with a side panel, under
the exact lockstep rules of the 13–3, for a 6-hour session on one g6e.2xlarge. A replay version is built as a fallback.

**Architecture:**
- **Live mode (A1):** the training `LockstepClock` gets a wall-time floor (`pace_s`). A new engine patch, `-rl-paced`,
  lets the rendered engine interpolate between turns while running only the turns `/step` allows.
- **Replay mode (A2):** games play headless on training's own engine image. The engine's visual replay then renders
  each game, with the overlay following the game's timeline.
- **Shared:** a session runner (`agentenv_zeroad.showcase`) plays the rotation, classifies each game, serves the
  overlay and keeps `session.json`. Shell scripts in `live/` launch, run, upload and power the box off.

**Tech Stack:** Python 3.12 asyncio, httpx, Starlette (already in through FastMCP), pytest with virtual time; bash
`live/` scripts tested by `tests/test_live_scripts.py`; a C++ patch to 0 A.D. 0.28 built in Docker; vLLM v0.11.0.

**Spec:** `docs/superpowers/specs/2026-11-19-showcase-stream-design.md` (approved 19 Nov 2026). Read it before any task.

## Global Constraints

- **The rules are the 13–3's.** Every showcase game spec comes from `live/s2/run1-frontier.toml` through
  `train/panels.py`, never re-typed:
  - latency 5000 ms;
  - decisions every 25 s, minimum gap 8 s;
  - interface v2, 30-minute cap, score rule `net_worth`;
  - `BRAIN_TIMEOUT_S` 120;
  - the policy samples with `TRAIN_SAMPLING` (`{"temperature": 1.0, "top_p": 1.0, "top_k": -1}`);
  - the teachers use the baselines' frontier-panel model, effort and max_tokens.
- **Pacing changes no game:**
  - `pace_s` defaults to `None`, and training never sets it;
  - the showcase sets 0.2 s per turn;
  - training's engine image (`zeroad-engine:0.28-rt`) is never rebuilt with the new patch.
- **vLLM is served as in Run 1:**
  - image `vllm/vllm-openai:v0.11.0`, flags `--dtype bfloat16 --generation-config vllm --enable-lora --max-lora-rank 32
    --max-model-len 8192`;
  - only `--gpu-memory-utilization` (0.55) and `--max-loras` (1) differ;
  - the adapter is Run 1's `adapters/v50`, served as `local/policy-v1`.
- **Fresh starts:** positions come from `[held_out]`, at index 400 or above. A seed any earlier config's panel used is
  never played, and a start is never played twice.
- **No result:**
  - when either side's failed calls (errors, not timeouts) exceed 10% of its decisions;
  - when the engine or the box fails;
  - when a side fails 5 times in a row (the game ends at once).
  - After two no-results in a row from one opponent's failures, that opponent is dropped.
  - Nothing is replayed.
- **On the box:** no command outside `live/` scripts. Every launching, changing or spending script prints its plan
  without `--yes`. Run it once without `--yes`, read the plan, then run it with `--yes`; never chain the two.
- **Secrets:**
  - the LiteLLM key is never printed (sha256 prefix and status codes only);
  - the bucket name is redacted in all output;
  - no key, endpoint, prompt or bucket appears in `state.json`, `session.json` or logs.
- **Streaming:** no stream leaves the box. `live/push.sh` is never called, and the only RTMP target is the box's
  loopback relay.
- **Copy:** the backstory card's text is fixed (Task 6). The header says "replay · played HH:MM ET" in replay mode.
- **Git:**
  - branch `v2-showcase`, stacked on `v2-s2` with `gh stack`;
  - push only with `gh stack push`, after the main session's leak grep over `git diff 059c1c2 HEAD` finds nothing
    (its patterns stay outside the repo);
  - commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Tests:** run only the task's own test files locally; CI runs the rest.

## Review Focus

1. **A key or vLLM is down when the session starts.** Expected: the runner refuses before the first game, so it doesn't
   play 11 no-results. Test: Task 7 `test_preflight_failure_refuses_to_start`.
2. **A session is stopped or crashes and is started again.** Expected: it continues from the next unused start and
   never replays one. Tests: Task 7 `test_restart_continues_from_next_index`, Task 4
   `test_used_seeds_include_earlier_sessions`.
3. **vLLM errors on the shared GPU mid-game.** Expected: they count as failed calls for Qwen, the same as proxy errors
   for Opus, while a timeout never does. Tests: Task 5 `test_vllm_errors_count_as_failed_calls`,
   `test_timeouts_are_not_failed_calls`.
4. **The power-off deadline falls inside a game.** Expected: no game starts with less than 45 minutes left, and the
   upload always runs before the power-off. Tests: Task 7 `test_poweroff_margin_ends_session`, Task 10
   `test_showcase_run_uploads_before_shutdown`.
5. **The overlay leaks a secret or a prompt.** Expected: `state.json` carries none. Test: Task 6
   `test_state_has_no_secrets`.

## Execution order

| Wave | Tasks | Unblocks |
|---|---|---|
| 1 (parallel, own worktrees) | 1, 2, 4, 5, 6, 9 | the box can be launched and bootstrapped once 2 and 9 land |
| 2 | 3, then 7 and 10 | the spike (stage 1) |
| 3 | 8 | A2's proof on the spike game's replay |
| Stages | 11 | the test game, the rehearsal and the session |

Tasks 9 and 10 both touch `tests/test_live_scripts.py`, so 10 rebases onto 9.

---

### Task 1: The paced lockstep clock

**Files:**
- Modify: `src/agentenv_zeroad/clock.py` (`LockstepClock.__init__`, `run`, `_settle`, `metrics`)
- Test: `tests/test_lockstep_clock.py`

**Interfaces:**
- Produces:
  - `LockstepClock(..., pace_s: float | None = None)`. `pace_s` must be `None` or above 0; otherwise `ValueError`.
  - `LockstepClock.holding: dict[str, Any] | None`, set to `{"slot": str, "since": float}` (loop time the block began)
    while `_settle` waits on a seat, and `None` otherwise.
  - `metrics()` adds `"pace_s"` and `"paced_wait_s"` (total wall seconds slept for pacing).
- **Rule:** before each `step_once`, sleep until `next_due`, then set `next_due = max(now, next_due) + pace_s`. Steps
  are never closer than `pace_s`, and never burst to catch up after a hold.

- [ ] **Step 1: Write the failing tests** (virtual time from `tests/virtual_time.py`; fake brains as in this file)
  - `test_paced_clock_steps_no_faster_than_pace`: 50 steps, `pace_s=0.2`, an instant engine. The virtual elapsed time
    is at least 9.8 s, and every gap between step starts is at least 0.2 s.
  - `test_unpaced_clock_never_sleeps_for_pace`: with `pace_s=None`, `metrics()["paced_wait_s"] == 0`, and an instant
    engine with an instant brain ends at virtual time 0.
  - `test_paced_clock_does_not_burst_after_hold`: one seat answers its first decision in 3.0 s (wall). After the block,
    the next 10 step-start gaps are all at least 0.2 s.
  - `test_holding_names_the_slot_during_a_block_and_clears_after`: sampled while blocked,
    `holding == {"slot": "1", "since": <the block's start>}`; `None` after.
  - `test_pacing_changes_nothing_a_seat_sees`: two seats, with deterministic fake decisions answering in 0.5, 4.0, 6.5
    and 130 s (one beyond `BRAIN_TIMEOUT_S`). Run 600 steps paced (0.2) and unpaced. `clock.commands`,
    `clock.decisions` (`frozen_ms`, `due_ms`, `applied_ms`, `missed`) and the `game_ms` every `begin` saw are equal.
  - `test_pace_must_be_positive`: `pace_s=0` and `pace_s=-1` raise `ValueError`.

- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_lockstep_clock.py -q`. Expected: the new
  tests fail (`TypeError: unexpected keyword 'pace_s'`).
- [ ] **Step 3: Implement** `pace_s`, `holding` and the metrics in `clock.py`. Pacing uses `self._sleep` and `self._now`
  (the virtual clock in tests). `holding` is set and cleared inside `_settle`, around the `asyncio.wait`.
- [ ] **Step 4: Run the file's tests.** Expected: all pass, the old ones included.
- [ ] **Step 5: Commit** `Add a wall-time floor to the lockstep clock`.

### Task 2: The paced render engine

**Files:**
- Create: `patches/rl-paced-render.patch`, `engine/Dockerfile.show`, `tests/engine/test_paced_engine.py`
- Modify: `scripts/build-engine.sh` (a `show` target), `src/agentenv_zeroad/engine.py` (`EngineOptions`,
  `pyrogenesis_args`)
- Test: `tests/test_engine_launch.py`, `tests/engine/test_paced_engine.py`

**Interfaces:**
- Produces:
  - `EngineOptions.paced: bool = False`;
  - `pyrogenesis_args` appends `-rl-paced` when `paced`, and raises `ValueError` when `paced` is set with `realtime`
    or without `visual`;
  - the image `zeroad-engine:0.28-show`, built by `scripts/build-engine.sh show` and labelled
    `zeroad.patches="rl-realtime rl-idle-wait rl-paced-render"`.
- **The patch's contract** (visual mode with `-rl-paced` only):
  1. A reset game stays at turn 0 until the first `/step`.
  2. Each `/step` allows exactly one more 200 ms turn. The frame loop runs it (`g_Game->Update` with the turn manager
     capped at the turns allowed) and keeps interpolating. `/step` answers after that turn has run, with the state, or
     `{}` for `?state=0`.
  3. With no `/step` pending, sim time doesn't move. The frame loop still renders, and the camera and the director
     still work. Units hold at the end of the last turn (`CTurnManager::Interpolate` clamps its offset).
  4. Without `-rl-paced`, the binary behaves as the rt image's.
- **The patch is layered:** it applies on top of `rl-realtime.patch` and `rl-idle-wait.patch` with `patch -p1 -F0`.
  `Dockerfile.show` repeats `Dockerfile.rt`'s lines up to its `COPY` (so the build-dep layer is cached), then applies
  all three patches.

- [ ] **Step 1: Write the failing launch tests** in `tests/test_engine_launch.py`:
  - `test_paced_engine_args`: `"-rl-paced" in pyrogenesis_args(EngineOptions(visual=True, display=":0", paced=True))`
    and `"-rl-realtime" not in` it.
  - `test_paced_needs_visual_and_not_realtime`: `paced=True` without `visual`, and with `realtime=True`, both raise
    `ValueError`.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_engine_launch.py -q`.
- [ ] **Step 3: Implement** `EngineOptions.paced` and the argument check. Run the tests and see them pass.
- [ ] **Step 4: Write the engine test** `tests/engine/test_paced_engine.py`. Mark it `engine`, and skip it when the
  docker image `zeroad-engine:0.28-show` is missing. Visual on Xvfb inside the container, as
  `tests/engine/test_rt_engine.py` does it.
  - `test_paced_engine_holds_until_stepped`: after `/reset`, sim time (`sim_time_ms()`) is 0 after 3 s of wall time.
  - `test_each_step_is_one_turn`: 25 steps give exactly 5000 ms.
  - `test_no_turns_between_steps`: after 10 steps, a 2 s pause leaves sim time at 2000 ms.
  - `test_paced_view_is_the_observers`: the director mod logs player −1, as in the rt test.
- [ ] **Step 5: Write the patch, `Dockerfile.show` and the `show` target.** Build them on any amd64 docker host
  (`scripts/build-engine.sh show`, about 15 minutes; on the box, Task 9's bootstrap builds them).
- [ ] **Step 6: Run the engine test against the image.** Run
  `uv run pytest -m engine tests/engine/test_paced_engine.py -q`. Expected: 4 pass. Where no amd64 docker host
  exists locally, this step runs on the box as part of Task 9's bootstrap check, and the task isn't done until it
  passes there.
- [ ] **Step 7: Commit** `Add the paced render engine patch and image`.

### Task 3: A paced lockstep game the runner can watch

**Files:**
- Modify: `src/agentenv_zeroad/train/game.py` (`LockstepMatch.__init__`, `play_game`), `src/agentenv_zeroad/live.py`
  (`Broadcast.attach_game`)
- Test: `tests/test_lockstep_match.py`, `tests/test_live.py`

**Interfaces:**
- Consumes: Task 1's `LockstepClock(pace_s=...)`.
- Produces:
  - `LockstepMatch(handle, config, *, latency_ms, brain_timeout_s, pace_s: float | None = None, **kwargs)` passes
    `pace_s` to its clock.
  - `play_game(..., pace_s: float | None = None, on_match: Callable[[LockstepMatch], None] | None = None)` calls
    `on_match(match)` once, after `setup()` and before `start()`.
  - `Broadcast.attach_game() -> None` (async). It runs `window_move_command` and `window_focus_command` for the
    engine window that just opened. The session owns one `Broadcast` and calls this once per game. The match is never
    given the broadcast, so ending a match never stops the session's recording.

- [ ] **Step 1: Write the failing tests:**
  - `test_lockstep_match_passes_pace_to_its_clock`: `match.lockstep.metrics()["pace_s"] == 0.2`.
  - `test_play_game_hands_the_match_over_before_start`: `on_match` sees `match.started is False`, and is called once.
  - `test_play_game_unpaced_by_default`: `pace_s` is `None` in the results' clock metrics.
  - In `tests/test_live.py`, `test_attach_game_moves_and_focuses_the_new_window`: with xdotool stubbed as in the file's
    other broadcast tests, the two commands run in order.
- [ ] **Step 2: Run them and see them fail.** Run
  `uv run pytest tests/test_lockstep_match.py tests/test_live.py -q`.
- [ ] **Step 3: Implement.** Run them and see them pass.
- [ ] **Step 4: Commit** `Let a lockstep game be paced and watched`.

### Task 4: Showcase starts from the 13–3's config

**Files:**
- Create: `live/showcase/showcase.toml`, `src/agentenv_zeroad/showcase/__init__.py`,
  `src/agentenv_zeroad/showcase/specs.py`
- Modify: `src/agentenv_zeroad/train/panels.py` (add `game_at`; `frontier_specs` and `heldout_specs` keep their output
  byte for byte)
- Test: `tests/test_showcase_specs.py`

**Interfaces:**
- Produces:
  - `panels.game_at(cfg: RunConfig, panel: PanelConfig, j: int, opponent: str, base_url: str = UNSET_URL) -> GameSpec`.
    - `opponent` is a teacher name of `panel` or `"petra-<n>"`.
    - The position is `position(cfg, "held_out", j)`; the policy is `_policy`, the teacher `_teacher_seat`.
    - Petra is `SeatSpec(kind="petra", civ=pos.opponent_civ, difficulty=n, behaviour=pos.behaviour)`, as in
      `heldout_specs`.
    - Seats alternate by `j % 2`; `game_id = f"{panel.name}-{j:04d}-{opponent}"`.
  - `ShowcaseConfig.load(path) -> ShowcaseConfig` with `rules_config: Path`, `panel: str`, `rotation: tuple[str, ...]`,
    `first_index: int`, `adapter_version: int`.
  - `used_seeds(configs: Iterable[Path], sessions: Iterable[Path] = ()) -> set[int]`: every seed of every panel spec of
    every config (through `train.calibration.panel_specs`), plus every seed in the given `session.json` files.
  - `next_starts(cfg: RunConfig, first_index: int, used: set[int], count: int) -> list[int]`: indices `j >= first_index`
    whose position seed isn't in `used`, in order. It raises `ValueError` rather than pass
    `len(cfg.held_out.all_seeds())`, where positions wrap.
- **`live/showcase/showcase.toml`:** `rules_config = "live/s2/run1-frontier.toml"`, `panel = "frontier-final"`,
  `rotation = ["opus", "astra", "petra-3"]`, `first_index = 400`, `adapter_version = 1`.

- [ ] **Step 1: Write the failing tests:**
  - `test_game_at_matches_frontier_specs`: for opus and astra, j = 0..15, `game_at` equals `frontier_specs`' game j
    with `game_id` set aside.
  - `test_petra_game_matches_heldout_specs`: Petra-3 at j equals `heldout_specs`' game j for a held-out panel with
    `opponents = {"petra-3": 1}` on the same config, apart from `game_id`.
  - `test_teacher_settings_are_the_13_3s`: each teacher seat's `model`, `effort` and `max_tokens` equal the baselines'
    frontier panel's (`live/s2/baselines.toml`).
  - `test_policy_seat_samples_like_training`: `sampling == TRAIN_SAMPLING`, `model == "local/policy-v1"`,
    `interface == "v2"`, `trainable`.
  - `test_rules_are_the_13_3s`: `latency_ms == 5000`, `time_cap_minutes == 30`, `score_rule == "net_worth"`, and every
    LLM seat has `interval_s == 25` and `min_gap_s == 8`.
  - `test_showcase_seeds_avoid_every_earlier_panel`: `next_starts` over `used_seeds(live/s2/*.toml)` returns no seed in
    that set.
  - `test_used_seeds_include_earlier_sessions`: a `session.json` holding seed 900401 removes it from `next_starts`.
  - `test_starts_never_wrap`: asking for more starts than are left raises `ValueError`.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_showcase_specs.py -q`.
- [ ] **Step 3: Implement** `game_at`, `ShowcaseConfig`, `used_seeds`, `next_starts` and the TOML. Run the new file,
  plus `tests/test_s2.py -k frontier`, which pins `frontier_specs`. Expected: all pass.
- [ ] **Step 4: Commit** `Add showcase starts from the 13-3's config`.

### Task 5: Results, checks and the session file

**Files:**
- Create: `src/agentenv_zeroad/showcase/results.py`
- Test: `tests/test_showcase_results.py`

**Interfaces:**
- Produces:
  - `GameRow` (a frozen dataclass): `index: int`, `game_id: str`, `opponent: str`, `seed: int`, `policy_slot: str`,
    `result: Literal["win", "loss", "draw", "no_result"]`, `how: Literal["conquest", "net_worth"] | None`,
    `reason: str | None`, `failed: dict[str, int]`, `decisions: dict[str, int]`, `played_at: float`,
    `problems: list[str]`. It has `to_json()` and `from_json()`.
  - `classify(results: Mapping[str, Any], *, index: int, opponent: str, policy_slot: str) -> GameRow`:
    - `"no_result"` when `results["failed"]` is set (engine or box);
    - `"no_result"` when `results.get("showcase", {}).get("ended_early")` is set (5 failures in a row);
    - `"no_result"` when a seat's failed calls are more than 10% of its decisions;
    - otherwise the policy seat's outcome as grading wrote it (`won` → win, `lost` → loss, `drawn` → draw), with
      `how` from how the game ended.
    - Failed calls are `results["sampler"][slot]["failed_calls"]`. Petra has no calls. A timeout cancels the call, and
      the decision is missed, but it is not a failed call.
    - The reason names the model and the count: `"Astra, 41 failed calls"`.
  - `verify_lockstep(results: Mapping[str, Any]) -> list[str]`: the problems found.
    - Every applied decision must have `applied_ms - frozen_ms == 5000`.
    - The spec must have latency 5000, a 30-minute cap, `net_worth`, interface v2 and `TRAIN_SAMPLING` for the policy.
  - `Session(path: Path)`:
    - `load()`, and `save()` (atomic, as `train/s2.py` `Record._write`);
    - `rows: list[GameRow]`;
    - `add(row)`;
    - `table() -> list[dict]`: one row per opponent in rotation order, `{"opponent", "w", "l", "d", "nr"}`;
    - `dropped() -> set[str]`: opponents whose last two games were no-results from their own failures;
    - `next_index() -> int`;
    - `seeds() -> set[int]`;
    - `ended: str | None`, and why.

- [ ] **Step 1: Write the failing tests** (results dicts built from the fields above):
  - `test_win_by_conquest` and `test_win_by_net_worth_at_cap`;
  - `test_exact_tie_is_a_draw`;
  - `test_engine_failure_is_no_result`;
  - `test_ten_percent_failed_calls_is_no_result_either_side`: 3 of 20 for Opus gives no result; 2 of 20 counts; the
    same for the policy;
  - `test_vllm_errors_count_as_failed_calls`: the policy seat's `failed_calls` of 5 in 30 gives no result;
  - `test_timeouts_are_not_failed_calls`: 4 missed decisions with `failed_calls == 0` counts;
  - `test_ended_early_is_no_result_with_reason`;
  - `test_verify_lockstep_flags_an_order_off_t_plus_5`, and `test_verify_lockstep_passes_a_13_3_game`, using a
    `results.json` fixture cut from a run1-frontier game in `tests/fixtures/` with every prompt removed;
  - `test_table_in_rotation_order_counts_no_results`;
  - `test_two_no_results_in_a_row_drop_an_opponent`, and `test_no_results_from_the_policy_do_not_drop_the_opponent`;
  - `test_session_round_trips_and_next_index_follows_the_last_row`.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_showcase_results.py -q`.
- [ ] **Step 3: Implement.** Run them and see them pass.
- [ ] **Step 4: Commit** `Add showcase results, lockstep checks and the session file`.

### Task 6: The overlay blocks and their server

**Files:**
- Create: `src/agentenv_zeroad/showcase/overlay.py`
- Modify: `src/agentenv_zeroad/viewer/overlay.html` (new blocks, drawn only when their keys are present)
- Test: `tests/test_showcase_overlay.py`

**Interfaces:**
- Consumes: Task 5's `Session`, `live.overlay_state`.
- Produces:
  - `BACKSTORY: tuple[str, ...]`, exactly these lines:
    - `"RUN 1: 8 h on one box"`
    - `"vs Petra  37.8% → 72.4% won"`
    - `"vs Opus 13-3 · Astra 9-7"`
    - `"43.7M of 8.2B weights trained (0.5%)"`
    - `"50 updates · 1,462 games"`
  - `showcase_state(session, match, *, phase, game_no, round_no, hold, mode, played_at, intermission) -> dict[str, Any]`
    is `overlay_state(match)` (or `{}` with no match) plus:
    - `series`: `{"policy": "QWEN3-8B + 8 h RL", "opponent", "game", "round", "clock"}`;
    - `session`: `session.table()`, plus the latest no-result's reason;
    - `backstory`: `BACKSTORY`;
    - `hold`: `{"model", "waited_s"}` or `None`;
    - `intermission`: `{"last": {"winner", "how"}, "next", "table"}` or `None`;
    - `replay`: `{"played_at": "14:05 ET"}` or `None`.
  - `hold_of(match) -> dict | None`: from `match.lockstep.holding`, with the seat's model label from `overlay_state`'s
    `model` field and `waited_s` to 0.1 s.
  - `OverlayServer(state: Callable[[], dict[str, Any]], token: str, host: str = "127.0.0.1", port: int = 18766)`, with
    `start()` and `stop()`. A Starlette app with uvicorn on loopback:
    - `GET /live` returns `overlay_html()`;
    - `GET /live/state.json` refuses a wrong token or a non-loopback client with 403, as `server.py` `_live_state`.
- **Page:**
  - The hold line reads `⏸ waiting on <model> · <s> s`.
  - The intermission card replaces the seat panels while `intermission` is set.
  - With no showcase keys, the env's overlay looks as it does today.

- [ ] **Step 1: Write the failing tests:**
  - `test_state_has_no_secrets`: the state's JSON for a match whose seats have a `base_url`, a key in the environment
    and a prompt contains none of them, and no bucket-shaped `s3://` string.
  - `test_hold_line_names_the_model_and_seconds`: holding slot "2" (Opus) since 6.2 s ago gives
    `{"model": "Opus", "waited_s": 6.2}`.
  - `test_intermission_replaces_panels`.
  - `test_table_rows_in_rotation_order_with_no_result_reason`.
  - `test_replay_label`.
  - `test_backstory_copy`: `BACKSTORY` equals the five lines above.
  - `test_state_json_refuses_wrong_token_and_non_loopback`, with Starlette's `TestClient`.
  - `test_page_draws_showcase_blocks_only_when_present`: the page source branches on `state.series`, `state.hold` and
    `state.intermission`, with the existing structure kept (a string check, like `tests/test_live.py`'s page tests).
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_showcase_overlay.py -q`.
- [ ] **Step 3: Implement.** Run them, plus `tests/test_live.py`. Expected: all pass.
- [ ] **Step 4: Commit** `Add the showcase overlay blocks and server`.

### Task 7: The session runner (live mode)

**Files:**
- Create: `src/agentenv_zeroad/showcase/session.py`, `src/agentenv_zeroad/showcase/__main__.py`
- Test: `tests/test_showcase_session.py`

**Interfaces:**
- Consumes: Task 3's `play_game(pace_s, on_match)` and `Broadcast.attach_game`; Task 4's specs; Task 5's `classify`,
  `verify_lockstep` and `Session`; Task 6's `showcase_state`, `hold_of` and `OverlayServer`; Task 2's
  `EngineOptions(paced=True)`.
- Produces:
  - `SessionOptions` (a dataclass):
    - `out: Path`, `hours: float = 6.0`, `max_games: int | None = None`, `rotation: tuple[str, ...]`;
    - `mode: Literal["live", "replay"] = "live"`, `pace_s: float = 0.2`, `intermission_s: float = 120.0`;
    - `min_poweroff_margin_s: float = 2700.0`.
  - `SessionDeps` (a dataclass of callables, faked in tests):
    - `start_engine() -> RunningEngine`;
    - `budget_ok(planned_usd: float, planned_hours: float) -> Awaitable[bool]` (runs `budget.sh check`);
    - `poweroff_left_s() -> float`;
    - `model_for(seat, slot) -> StrategistModel`;
    - `preflight() -> Awaitable[list[str]]`: problems, one call per model in the rotation;
    - `now() -> float`, `sleep(s) -> Awaitable[None]`.
  - `async def run_session(opts: SessionOptions, cfg: ShowcaseConfig, deps: SessionDeps, broadcast: Broadcast | None)
    -> Session`.
  - **CLI:**
    - `python -m agentenv_zeroad.showcase run --out DIR --config live/showcase/showcase.toml [--hours H]
      [--max-games N] [--rotation a,b,c] [--mode live|replay]`;
    - `python -m agentenv_zeroad.showcase verify DIR` prints each game's problems and exits 1 if there are any.
  - **The CLI's real dependencies:**
    - `start_engine` is `engine.start_engine(EngineOptions(port=free_port(), visual=True, display=":0", paced=True,
      extra_args=live.engine_args(low_graphics=False)), mode="local", ready_timeout=180)`.
    - One `Broadcast` runs for the session: `BroadcastSettings(display=":0", encoder="nvenc", low_graphics=False,
      rtmp_url=<the ZEROAD_RTMP_URL the docker shim sets, the loopback relay>)`. It's started once on the
      `OverlayServer`'s URL, and stopped when the session ends.
    - `model_for` is `build_model` with the same `cap_usd` the run1-frontier games' seats had (read it from
      `train/orchestrate.py`'s frontier path; $6 if they had none). The budget check plans that cap per frontier
      seat.
- **Each game:**
  1. Check the budget for the opponent's seat caps and 0.75 h.
  2. Check that the power-off is at least `min_poweroff_margin_s` away.
  3. Start the paced engine on the display, then `broadcast.attach_game()`.
  4. Run `play_game(pace_s=opts.pace_s, on_match=...)` with a failure watch: it reads each seat's last 5 calls every
     second and stops the match with `ended_early` when all 5 failed.
  5. `classify` the game and `verify_lockstep` it. Add the row and save.
  6. Show the intermission for `intermission_s`, with the finished engine left running and unstepped, so its last frame
     shows.
  7. Stop the engine.
- **Session rules:**
  - The session ends at the first game boundary after `hours`, or after `max_games`. The reason goes in
    `session.ended`.
  - SIGTERM stops the game in play, records it as no result ("stopped by hand") and ends the session.
  - The overlay's state is `showcase_state(...)` over the current match.

- [ ] **Step 1: Write the failing tests** (`FakeEngine`, `tests/fake_models.py`, virtual time):
  - `test_session_plays_rotation_in_order`: opus, astra, petra-3, opus.
  - `test_session_ends_after_hours_at_a_game_boundary`.
  - `test_max_games`.
  - `test_budget_refusal_ends_session_before_a_game`.
  - `test_poweroff_margin_ends_session`.
  - `test_engine_crash_is_no_result_and_rotation_moves_on`.
  - `test_five_failures_in_a_row_end_game_early`.
  - `test_dropped_opponent_is_skipped`.
  - `test_restart_continues_from_next_index`: a second `run_session` on the same `out` starts at the next unused index,
    and its seeds aren't in the first run's.
  - `test_preflight_failure_refuses_to_start`: no game is played, and `session.ended` names the failing model.
  - `test_sigterm_records_stopped_by_hand`.
  - `test_games_are_paced`: the clock metrics in each `results.json` show `pace_s == 0.2`.
  - `test_verify_cli_exits_1_on_a_problem`.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_showcase_session.py -q`.
- [ ] **Step 3: Implement.** Run them and see them pass.
- [ ] **Step 4: Commit** `Add the showcase session runner`.

### Task 8: Replay mode (A2)

**Files:**
- Create: `src/agentenv_zeroad/showcase/replay.py`
- Modify: `src/agentenv_zeroad/showcase/session.py` (`mode="replay"`)
- Test: `tests/test_showcase_replay.py`

**Interfaces:**
- Consumes: Task 7's `run_session`, `SessionDeps` and `Session`; Task 6's `showcase_state`.
- Produces:
  - `replay_args(commands: Path, game: tuple[int, int] = GAME) -> list[str]`: pyrogenesis arguments for the engine's
    visual replay of `commands` (`-replay-visual=`, checked against 0.28 in the spike), plus `live.engine_args(...)`.
    There's no RL interface.
  - `ReplayTimeline(results: Mapping[str, Any])` with `state_at(game_ms: int) -> dict[str, Any]`, shaped like
    `overlay_state`'s `seats` and `scoreboard`. The values come from `results["timeline"]` (10 s rows) and orders from
    each seat's turns with `game_ms <= game_ms`. There's no `hold` and no `thinking_s`.
  - In replay mode, `run_session`:
    - plays each game headless on training's image (`pool.TRAIN_IMAGE`, no pacing), exactly the 13–3's path;
    - copies `commands.txt` beside `results.json`;
    - renders the replays in the order played, at most one game played ahead of the one on screen;
    - adds the row to the table only when that game's replay ends, so the table never shows a result before the
      picture does.
  - **Game time during a replay:** the spike picks the source. Requirement: the overlay clock stays within 1 s of the
    game's own clock over a whole game, measured on the spike recording.

- [ ] **Step 1: Write the failing tests:**
  - `test_timeline_state_at_matches_results`.
  - `test_recent_orders_never_from_the_future`.
  - `test_replay_mode_shows_every_game_in_order_played`: none skipped, no-results included.
  - `test_table_updates_only_when_replay_ends`.
  - `test_header_says_replay_and_time_played`: `"replay · played 14:05 ET"` from an epoch in ET.
  - `test_replay_args_include_director_mod_and_replay_path`.
  - `test_headless_games_use_the_training_image_unpaced`.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_showcase_replay.py -q`.
- [ ] **Step 3: Implement.** Run them, plus `tests/test_showcase_session.py`. Expected: all pass.
- [ ] **Step 4: Commit** `Add replay mode to the showcase`.

### Task 9: The showcase box

**Files:**
- Create: `live/showcase.env.example`, `live/box/showcase-bootstrap.sh`
- Modify: `live/bootstrap.sh` (`--showcase`), `.gitignore` if `live/showcase.env` isn't covered
- Test: `tests/test_live_scripts.py`

**Interfaces:**
- **`live/showcase.env.example`:**
  - sources `live/live.env`;
  - `LIVE_GPU_TYPES="g6e.2xlarge"`, `LIVE_GPU_RATES="g6e.2xlarge=2.24"`;
  - `LIVE_MAX_HOURS=10`, `LIVE_BUDGET_USD=60`, `LIVE_SHUTDOWN_BEHAVIOR=stop`;
  - the training box's instance profile, under the variable `live/s2.env.example` uses.
- **`live/bootstrap.sh ID --showcase [--yes|--dry-run]`** runs `box/showcase-bootstrap.sh PAYLOAD_URL REPO_URL` through
  SSM. That script:
  1. checks the GPU (at least 40 GB, one GPU);
  2. does `box/bootstrap.sh`'s stream setup: Xorg, ffmpeg, mediamtx, uv, the units, the display and the relay;
  3. builds `zeroad-engine:0.28-show` (`scripts/build-engine.sh show`), then the env and live images on it, tagged
     `zeroad-showcase` (the manual steps in `streamer/Dockerfile`, with `ENGINE_IMAGE=zeroad-engine:0.28-show`);
  4. starts vLLM (one instance, the Global Constraints' flags, its key made in a root-only file as
     `pilot-bootstrap.sh` does);
  5. copies `s3://$LIVE_BUCKET/zeroad-train/run/s2-run1/adapters/v50/` to `$STATE/showcase/adapters/v1` and checks it
     against the files' sizes and checksums in S3;
  6. loads the adapter as `local/policy-v1` and makes one strategist call, which must come back with a tool call and
     token ids;
  7. runs `pytest -m engine tests/engine/test_paced_engine.py` in the showcase image;
  8. prints the GPU memory used.

  It is idempotent. Shared steps are factored into `box/lib.sh` functions instead of copied, where `bootstrap.sh` and
  `pilot-bootstrap.sh` already do them.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_scripts.py`, in the file's harness style:
  - `test_showcase_bootstrap_dry_run_prints_the_plan`: the vLLM flags as pinned, the show image and the adapter path,
    with the bucket redacted.
  - `test_showcase_bootstrap_refuses_a_small_gpu`.
  - `test_showcase_and_train_are_exclusive`.
  - `test_showcase_vllm_flags_match_run1_but_memory`: the flag set equals `--train`'s, apart from
    `--gpu-memory-utilization` and `--max-loras`.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_live_scripts.py -k showcase -q`.
- [ ] **Step 3: Implement.** Run them and see them pass, and run `shellcheck` on the changed scripts.
- [ ] **Step 4: Commit** `Add the showcase box bootstrap`.

### Task 10: The showcase scripts

**Files:**
- Create: `live/showcase.sh`, `live/box/showcase.sh`, `live/box/showcase-run.sh`
- Test: `tests/test_live_scripts.py`

**Interfaces:**
- **`live/showcase.sh ID start [--hours H] [--max-games N] [--rotation LIST] [--mode live|replay] | status | stop |
  fetch [--with-video] | watch [--yes|--dry-run]`.** Start, stop and fetch are dry runs unless `--yes`; status and
  watch run unless `--dry-run`.
  - **start** refuses:
    - when `LIVE_BUDGET_USD` would be passed by the box hours so far plus `H` at the box rate, plus the frontier games'
      caps;
    - or when the power-off leaves less than `H + 1` hours (`extend-poweroff.sh` moves it).
  - **status:** the unit, the game in play, the session table, the spend, the power-off and the GPU memory.
  - **stop:** SIGTERM. The game in play is recorded as no result, the session uploads, and the box stays up.
  - **fetch:** reads `s3://$LIVE_BUCKET/zeroad-train/run/showcase-<stamp>/` (no video unless `--with-video`) into
    `live/fetched/showcase-<stamp>/`.
  - **watch:** an SSM port-forward of the relay's RTMP port 1935 to the laptop's loopback, and prints
    `rtmp://127.0.0.1:1935/zeroad` to open in VLC or ffplay.
- **`live/box/showcase.sh start|status|stop`** runs the transient unit `zeroad-showcase`, whose process is
  `showcase-run.sh`.
- **`showcase-run.sh`:**
  1. runs `python -m agentenv_zeroad.showcase run` in the `zeroad-showcase` container, on display :0 with the GPU and
     the relay (as the docker shim does), under `with-model.sh` for the LiteLLM key, with vLLM's key from its
     root-only file;
  2. when the run ends, however it ends, runs `aws s3 cp` of the session folder and the relay segments written since
     the session started to `s3://$LIVE_BUCKET/zeroad-train/run/showcase-<stamp>/`;
  3. then `shutdown -h now`, unless `--no-shutdown` was given or the upload failed.

- [ ] **Step 1: Write the failing tests:**
  - `test_showcase_start_dry_run_prints_the_plan`;
  - `test_showcase_start_refuses_over_budget`;
  - `test_showcase_start_refuses_short_poweroff`;
  - `test_showcase_run_uploads_before_shutdown`;
  - `test_showcase_run_no_shutdown_when_upload_fails`;
  - `test_showcase_output_redacts_the_bucket`;
  - `test_showcase_never_pushes`: none of the three scripts mentions `push.sh`, and the only RTMP URL is the loopback
    relay's.
- [ ] **Step 2: Run them and see them fail.** Run `uv run pytest tests/test_live_scripts.py -k showcase -q`.
- [ ] **Step 3: Implement.** Run them and see them pass, and run `shellcheck` on the changed scripts.
- [ ] **Step 4: Commit** `Add the showcase scripts`.
- [ ] **Step 5: Push.** Run the leak grep, then `gh stack push`, and wait for CI. Every task's tests must be green in
  CI before the box runs this code.

### Task 11: The stages on the box

The stages from the spec's order of work, each with its own pass checks. The user's go-ahead is asked for at the time
for every AWS write. Every script runs once without `--yes`, is read, then runs with `--yes`.

- [ ] **Launch and bootstrap** (go-ahead: the launch):
  - `LIVE_ENV=live/showcase.env live/launch.sh --name zeroad-showcase --shutdown-behavior stop`, read the plan, then
    run it with `--yes`;
  - then `live/bootstrap.sh $ID --showcase`, read the plan, then run it with `--yes`.
  - Passes when the bootstrap's smoke call, adapter check and paced engine tests pass.
- [ ] **Stage 1, the spike** (v50 vs Petra-3, $0 model spend):
  - Run `live/showcase.sh $ID start --max-games 1 --rotation petra-3 --hours 1 --mode live`, read the plan, then run it
    with `--yes`.
  - Fetch the game with `fetch --with-video` and watch it.
  - Passes when all of these hold:
    - Qwen and the renderer fit on the GPU;
    - the game renders under the paced clock;
    - the director camera moves;
    - the overlay updates;
    - 30 fps (the engine log);
    - units move smoothly;
    - `verify` finds no problems;
    - the result on screen matches `results.json`.
  - Then render the same game in replay mode, and check stage 1's points plus the overlay clock (within 1 s). If live
    mode fails and replay mode passes, stages 3–5 run in replay mode.
- [ ] **Stage 3, the test game vs Opus** (go-ahead: restarting the box, then `extend-poweroff.sh`): `--max-games 1
  --rotation opus`. Passes when all of these hold:
  - the hold line appears while Opus is late and clears when its reply lands;
  - `verify` finds every order at T + 5 s;
  - the result on screen matches `results.json`;
  - the spend is close to $2.45.
- [ ] **Stage 4, the rehearsal:** `--hours 2`, one full round. Passes when all of these hold:
  - it runs to the end with nothing done by hand;
  - the session table matches every `results.json`;
  - the upload and then the power-off happen;
  - `fetch --with-video` brings back segments that play.
- [ ] **Stage 5, the session** (go-ahead: restarting the box, and the user's go): `--hours 6`.
- [ ] **Teardown** when the user says the showcase is done: `live/teardown.sh $ID --yes`. Confirm no `zeroad-showcase`
  instance is left, and report the box hours and the cost.
