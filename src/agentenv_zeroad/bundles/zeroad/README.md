# zeroad bundle

0 A.D. 0.28 matches in real time on agentenv-game-env (lobby `urn:game:lobby/v1`, match `urn:game:match/v1`).
Register the env first: `agent-env zeroad setup`.

| Task | Seats | Match | Cost |
|---|---|---|---|
| `smoke` | two scripted bots (Rome, Athens) | 128-tile Mainland, speed 2.0, 6 game-minutes | $0, about 3 min |
| `petra` | Petra against Petra (Persia, Carthage) | 128-tile Mainland, speed 2.0, 8 game-minutes | $0, about 4 min |
| `spike` | GPT-6 Astra + Luna reflex against Petra: latency in a live match | 128-tile Mainland, speed 1.0, 5 game-minutes | capped at $1.30 |
| `setup-a` | Opus 5.5 + Luna reflex against GPT-6 Astra + Luna reflex | 128-tile Mainland, speed 1.0, 10 game-minutes | capped at $2.40 |
| `setup-b` | GPT-6 Luna alone against Opus 5.5 alone (no reflexes) | 128-tile Mainland, speed 1.0, 10 game-minutes | capped at $1.50 |
| `livestream` | Setup A's lineup, rendered, directed and recorded (env `zeroad-live`) | 128-tile Mainland, speed 1.0, 40 game-minutes | capped at $9 (two strategists at $4, two reflexes at $0.50) |
| `livestream-4` | Opus 5.5, Sonnet 5.5, GPT-6 Astra and GPT-6 Sol, each with a Luna reflex, rendered (env `zeroad-live`) | 128-tile Mainland, speed 1.0, 10 game-minutes | capped at $5.40 (four strategists at $1.20, four reflexes at $0.15) |
| `livestream-4-long` | the same four seats as `livestream-4` | 128-tile Mainland, speed 1.0, 40 game-minutes | capped at $17.40 (four strategists at $4.20, four reflexes at $0.15); needs `LIVE_MATCH_CAP_USD=18` |
| `livestream-4-conquest` | the same four seats, until one empire is left | 128-tile Mainland, speed 1.0, a 12-hour time cap as a backstop | capped at $408 as a runaway guard (four strategists at $100, four reflexes at $2); needs `LIVE_MATCH_CAP_USD=410` and `LIVE_MAX_HOURS=14` |
| `livestream-petra` | the scripted bot against Petra, rendered and recorded (env `zeroad-live`) | 128-tile Mainland, speed 1.0, 5 game-minutes | $0, about 6 min |

The livestream tasks need the live env (`agent-env zeroad setup --live`). To change the lineup or the length, copy
the task and edit its slots or the lobby's `time_cap_minutes`.

The model tasks need the agents (`agent-env zeroad setup --agents`) and the model endpoint in agent-env's `[model]`
config, e.g. `base_url = "env:LITELLM_BASE_URL"`, `api_key = "env:LITELLM_API_KEY"` with both exported from a file
outside the repo. agent-env passes them to each agent as LITELLM_BASE_URL / LITELLM_API_KEY; the strategist hands
them to its seat's Luna reflex at run time (`POST /players/<id>/reflex`), since agent-env gives server envs no
environment of their own. Every brain stops before its cap (`ZEROAD_MAX_COST_USD` per agent, `reflex_max_usd` per
reflex seat). Before a paid run, check the loop's ledger and after it add the run's estimate:
`python -m agentenv_zeroad.spend check --ledger spend.jsonl --planned 5` and
`python -m agentenv_zeroad.spend add --ledger spend.jsonl --what "setup-a" --usd $(python -m agentenv_zeroad.spend results results.json)`.

## Settings
Lobby (`open_lobby.game_settings`): `map` (mainland), `size` (192), `seed` (7), `biome`, `placement`, `speed`
(1.0-2.0; 1.0 is real time), `time_cap_minutes` (40; 0 is no cap), `safety_cap_minutes` (45: where a game with no
time cap is decided anyway), `score_rule` (`official`, the engine's score total, or `net_worth`: the cost of a
player's living units and finished structures plus its banked resources; it decides the game at the time cap or safety
cap, conquest first; by default the rule the agents' prompt states, so `net_worth` when an agent plays interface `v2`,
and a rule that contradicts it fails the lobby), `draw_margin` (0 points of the rule's own measure), `ai_seed` (Petra's seed; the map seed by
default), `ceasefire_minutes` (0), `fog` (true), `start_grace_seconds` (30), `render` (false: render, direct and
record the match; needs the live env and speed 1.0), `rtmp` (false: with render, also push to the RTMP URL in the
env's `ZEROAD_RTMP_URL`). Experimental, names not yet checked against a running 0.28 engine:
`experimental_starting_resources`.

Player slot (`add_player_slot`): `player_id` is the engine's player number, `"1"` to `"8"`, without gaps.
`player_kind` is `agent` (a brain playing at `/players/<id>/mcp`) or `ai` with `game_settings.controller` `petra`
(default), `bot` (the scripted bot) or `replay` (a logged order stream: `replay_spec` is the path of a spec from
`scripts/s1_specs.py`). Also `civ`, `ai_difficulty` (Petra, 0-5), `ai_behavior` (Petra: `random`, `balanced`,
`aggressive` or `defensive`), `label`, `interface` (`v1` or `v2`), and for agent and bot seats `reflex: "luna"`
(GPT-6 Luna reflexes in the env, Setup A) with `reflex_max_usd`. Experimental, names not yet checked against a
running 0.28 engine: `experimental_resources` and `experimental_population_limit`.

## An agent seat (for model tasks)
```json
{"id": "agent-a", "type": "deploy_agent", "agent_name": "a", "a2a_agent_id": "<agent>", "env_ids": []},
{"id": "slot-1", "type": "add_player_slot", "env_id": "zeroad", "player_id": "1", "player_kind": "agent",
 "player_name": "a", "game_settings": {"civ": "rome"}, "depends_on": ["lobby", "agent-a"]},
{"id": "grade", "type": "grade_match", "env_id": "zeroad", "max_missed": 0.5, "depends_on": ["finish"]}
```
The agent plays through `get_brief` (`compact: true` for the JSON form) and one tool per order (`set_economy`, `train`, `build`, `research`,
`advance_phase`, `army`, `say`); each result is `{"status": "accepted" | "rejected" | "partial", "message", ...,
"footer"}`. Its brain reports each model call at `POST /players/<id>/brain_calls` with `{"seconds", "error"}` (and optionally
`usd`, `input_tokens`, `output_tokens`, `model`); a call that errors or takes over 120 s is missed, and `max_missed: 0.5` scores a seat that missed more than half its calls 0.
The match starts once every agent has called a tool, or after `start_grace_seconds`.
Beside it a brain posts `POST /players/<id>/waiting` `{"state": "begin" | "end"}` around a model call made while
the seat had a decision open (`waiting_share`), polls `GET /players/<id>/status` (running/over, game time, pending
and urgent events without draining them, and `open`: the decisions the seat can't act on until a brain answers), and
may hand its endpoint to the seat's reflex at `POST /players/<id>/reflex`. The Luna player reports a phase or attack
question it held past the plan's deadline at `POST /players/<id>/held` `{"decision": "phase" | "attack"}`; the
autopilot then carries out the default (advance to town; attack the leading enemy), logged as "autopilot (Luna held)".
A seat's `waiting_share` is measured
(0 when it never waited) once its brain polls status or reports a wait, or its reflex asks; otherwise it is null.

Agents (`agents/`, registered by `agent-env zeroad setup --agents`):
- `zeroad-commander`: a big model (the prompt step's `model`) as strategist: one call with the order tools every 25 s
  or on an urgent event, one in flight; OpenAI GPT-6 models on `/v1/responses` (reasoning effort low), others on
  `/v1/chat/completions`. Env: `ZEROAD_INTERVAL_S`, `ZEROAD_MIN_GAP_S`, `ZEROAD_MAX_COST_USD`,
  `ZEROAD_SHARE_REFLEX_KEY` (1).
- `zeroad-luna`: GPT-6 Luna alone: one Decisions request a second (`/openai/v1/decisions`) with strategic and
  tactical questions (economy, the next phase once it is nearly affordable, a timed attack, army size, what to train
  and build), answers mapped to order tools. Env: `ZEROAD_INTERVAL_S`, `ZEROAD_MAX_COST_USD`.

## Results
`save_match_files` keeps `results.json` (per seat: outcome, engine score, grade, forfeit, §5 metrics incl.
decisions per minute, latency p50/p95 and `waiting_share`, orders, tool call and brain call timings, `combat` (units
lost, enemy units killed, buildings lost), `actions`: the one record of who changed the seat's orders (`by`:
strategist, luna, reflex, autopilot or bot; the order as a tool call; its status; the overlay line, e.g. "pulled the
army home to defend the base"), with `action_counts`, the reflex's stats (requests, questions asked by name, `orders`
and `by_kind`) and the seat's estimated spend; `spend_usd_est` for the match) and the replay `commands.txt`; `save_zeroad_results` puts the results in the run's metadata
(`zeroad_results`).
