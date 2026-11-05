"""The post-training pilot without a box: its game tasks, the opponent ladder, how decisions are scored, the ratings
and the round driver's planning, collecting and progress page."""

import argparse
import json
import random
import statistics

from agentenv_zeroad.pilot import ladder, ratings, run, scoring
from agentenv_zeroad.pilot.games import GameSpec, Opponent, PilotSettings, petra, snapshot, task

SETTINGS = PilotSettings("http://172.17.0.1:8000", "box-token")


def spec(opp: Opponent, *, kind="train", slot="1", version=2) -> GameSpec:
    return GameSpec("r03-t01", 3, kind, 42, opp, version, policy_slot=slot, policy_civ="rome", opponent_civ="pers")


def steps_by_id(steps):
    return {s["id"]: s for s in steps}


def test_a_training_game_against_petra_logs_the_policys_turns_on_the_game_clock_at_temperature_one():
    steps = steps_by_id(task(spec(petra(3)), SETTINGS, "/pilot/r03/games/r03-t01"))
    env = steps["agent-policy"]["env_vars"]
    assert env["LITELLM_BASE_URL"] == "http://172.17.0.1:8000" and env["LITELLM_API_KEY"] == "box-token"
    assert env["ZEROAD_INTERVAL_CLOCK"] == "game" and env["ZEROAD_TURN_LOG"] == "1"
    body = json.loads(env["ZEROAD_EXTRA_BODY"])
    assert body["temperature"] == 1.0 and body["return_token_ids"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert steps["slot-2"]["game_settings"] == {"controller": "petra", "civ": "pers", "ai_difficulty": 3}
    assert steps["play-policy"]["model"] == "local/policy-v2" and "agent-opponent" not in steps
    assert steps["lobby"]["game_settings"]["speed"] == 2.0 and steps["lobby"]["game_settings"]["time_cap_minutes"] == 30
    assert steps["results"]["out_dir"] == "/pilot/r03/games/r03-t01"
    assert steps["finish"]["depends_on"] == ["play-policy"]


def test_a_snapshot_plays_on_vllm_unlogged_and_a_frontier_model_on_agent_envs_endpoint_with_a_cap():
    steps = steps_by_id(task(spec(snapshot(1), slot="2"), SETTINGS, "/o"))
    assert steps["play-opponent"]["model"] == "local/policy-v1" and steps["slot-1"]["player_name"] == "opponent"
    assert steps["agent-opponent"]["env_vars"]["ZEROAD_TURN_LOG"] == "0"
    frontier = Opponent("sonnet", "frontier", model="anthropic/claude-sonnet-5-5", trainable=False)
    steps = steps_by_id(task(spec(frontier, kind="probe"), SETTINGS, "/o"))
    assert "LITELLM_API_KEY" not in steps["agent-opponent"]["env_vars"]  # agent-env's own endpoint and key
    assert steps["agent-opponent"]["env_vars"]["ZEROAD_MAX_COST_USD"] == "5"
    assert steps["agent-policy"]["env_vars"]["ZEROAD_TURN_LOG"] == "0"  # a probe trains nothing
    assert json.loads(steps["agent-policy"]["env_vars"]["ZEROAD_EXTRA_BODY"])["temperature"] == 0.7
    assert steps["finish"]["depends_on"] == ["play-policy", "play-opponent"]


def test_round_one_plays_mostly_very_easy_and_easy_petra():
    pool = ladder.ladder([])
    picks = ladder.allot(16, pool, ladder.win_rates(pool, [], 0))
    counts = {o.id: sum(1 for p in picks if p.id == o.id) for o in pool}
    assert len(picks) == 16 and counts["petra-1"] + counts["petra-2"] >= 11 and counts["petra-5"] == 0


def test_the_mix_moves_up_as_the_policy_wins_and_keeps_a_share_for_what_it_mastered():
    pool = ladder.ladder([2])
    played = [ladder.Played("petra-1", r, 1.0) for r in range(3) for _ in range(4)]
    played += [ladder.Played("petra-2", r, 1.0) for r in range(3) for _ in range(3)]
    played += [ladder.Played("petra-3", 2, 0.5) for _ in range(4)]
    rates = ladder.win_rates(pool, played, 3)
    assert rates["petra-1"] > ladder.MASTERED_RATE and 0.4 < rates["petra-3"] < 0.6
    picks = [p.id for p in ladder.allot(16, pool, rates)]
    assert picks.count("petra-3") >= 5 and picks.count("v2") >= 4
    assert picks.count("petra-1") + picks.count("petra-2") >= 1  # the mastered share
    old = [ladder.Played("petra-3", 0, 0.0) for _ in range(20)]
    assert ladder.win_rates(pool, old, 9)["petra-3"] == ladder.PETRA_PRIOR[3]  # past the recent rounds


def test_the_probe_panel_and_its_seeds_stay_the_same_and_never_take_a_training_seed():
    frontier = run.frontier_list(run.FRONTIER)
    assert [(o.id, o.model) for o in frontier][1] == ("opus", "anthropic/claude-opus-5-5")
    panel = ladder.probe_panel(4, frontier)
    assert [o.id for o in panel] == [f"petra-{n}" for n in range(1, 6)] + ["v4", "sonnet", "opus", "astra", "sol"]
    seeds = [ladder.probe_seed(o) for o in panel]
    assert len(set(seeds)) == len(seeds) and min(seeds) > 9000
    assert ladder.allot(4, frontier, dict.fromkeys(["sonnet", "opus", "astra", "sol"], 0.5)) == []  # probes only
    turns = [[o.id for o in ladder.frontier_for(r, frontier, 1, 2)] for r in range(5)]
    assert turns == [["astra"], ["sol"], ["sonnet"], ["opus"], ["astra"]]  # one a round, in turn, from offset 2
    assert ladder.frontier_for(3, frontier, 0, 0) == frontier


def results(*, turns, totals_at, final, outcome="won"):
    timeline = [{"t_ms": t, "players": {"1": {"total": a, "pop": 10}, "2": {"total": b}}} for t, a, b in totals_at]
    return {
        "status": "finished",
        "ended_by": "time_cap",
        "game_time_s": 1800.0,
        "wall_s": 905.0,
        "clock": {"overruns": 2},
        "spend_usd_est": 0.0,
        "timeline": timeline,
        "seats": {
            "1": {
                "outcome": outcome,
                "score": {"total": final[0]},
                "moves": {"made": 3, "missed": 1},
                "turns": turns,
                "metrics": {"latency_s": {"p50": 3.2}},
            },  # fmt: skip
            "2": {"outcome": "lost", "score": {"total": final[1]}},
        },
    }


def turn(began, statuses=("accepted",)):
    orders = [{"tool": "train", "arguments": {}, "status": s} for s in statuses]
    return {"began_ms": began, "orders": orders, "prompt_token_ids": [1, 2], "token_ids": [3]}


def test_a_games_record_reads_the_policys_side_whichever_slot_it_played():
    res = results(turns=[turn(0, ("accepted", "rejected"))], totals_at=[(0, 0, 0)], final=(900, 400))
    rec = scoring.record(spec(petra(2)), res)
    assert rec.outcome == "won" and rec.margin == 500 and rec.result == 1.0 and rec.overruns == 2
    assert rec.calls == 4 and rec.missed == 1 and rec.rejected == 1 and rec.orders == 2 and rec.latency_p50_s == 3.2
    assert scoring.record(spec(petra(2), slot="2"), res).margin == -500
    assert scoring.record(spec(petra(2)), None, "no results.json").status == "missing"


def test_each_decision_carries_its_position_and_how_the_margin_moved_after_it():
    totals = [(0, 0, 0), (60_000, 100, 50), (120_000, 300, 100), (180_000, 500, 100)]
    res = results(turns=[turn(65_000), turn(130_000, ("rejected",))], totals_at=totals, final=(800, 200))
    first, second = scoring.decisions(spec(petra(2)), res)
    # margins 0, 50, 200, 400 at 0, 1, 2, 3 min; final 600. 65 s: from 50 to 400 by 3:05, 550 to the end.
    assert first.began_ms == 65_000 and first.short == 350 and first.long == 550
    assert second.short == 200 and second.long == 400 and second.rejected_share == 1.0
    assert first.features[1] == 1.0 and first.prompt_token_ids == [1, 2] and first.token_ids == [3]
    no_ids = {**turn(0), "token_ids": None}
    assert scoring.decisions(spec(petra(2)), results(turns=[no_ids], totals_at=totals, final=(1, 0))) == []


def test_a_decision_is_judged_against_what_its_own_position_predicted():
    rng = random.Random(0)
    batch, skill = [], []
    for i in range(200):
        ahead = rng.uniform(0, 2)
        skill.append(rng.gauss(0, 1))
        feats = [1.0, ahead] + [0.0] * 30
        batch.append(scoring.Decision("g", 0, 0, i, feats, 100 * ahead + 10 * skill[-1], 0.0, 0.0, [1], [2]))
    scoring.advantages(batch)
    adv = [d.advantage for d in batch]
    # being ahead explains most of each gain; the advantage is the part the decision added, not the lead
    assert statistics.correlation(adv, skill) > 0.95
    assert abs(statistics.correlation(adv, [d.features[1] for d in batch])) < 0.1
    rejected = scoring.Decision("g", 0, 0, 0, [1.0, 1.0] + [0.0] * 30, 100.0, 0.0, 1.0, [1], [2])
    accepted = scoring.Decision("g", 0, 0, 0, [1.0, 1.0] + [0.0] * 30, 100.0, 0.0, 0.0, [1], [2])
    scoring.advantages([rejected, accepted], batch)
    assert rejected.advantage < accepted.advantage and max(abs(d.advantage) for d in batch) <= scoring.CLIP


def test_ratings_order_the_players_and_put_petra_medium_at_1000():
    games = [("v1", "petra-3", 1.0)] * 8 + [("v1", "petra-3", 0.0)] * 2 + [("v0", "petra-3", 0.0)] * 8
    games += [("v0", "petra-3", 1.0)] * 2 + [("v1", "v0", 1.0)] * 3
    board = ratings.fit(games)
    assert board["petra-3"] == 1000 and board["v1"] > 1000 > board["v0"]
    assert ratings.fit([("v1", "petra-1", 1.0)] * 30)["v1"] < 3000  # unbeaten, but not infinite


def args(**over):
    base = {"train_games": 16, "probes": True, "frontier": "sonnet=anthropic/claude-sonnet-5-5"}
    base |= {"frontier_per_round": 0, "frontier_offset": 0}
    return argparse.Namespace(**{**base, **over})


def test_a_round_plans_its_training_games_and_the_same_probes_every_round():
    a = run.plan_round(run.State(round=3, version=2), [], args(), random.Random(1))
    b = run.plan_round(run.State(round=4, version=3), [], args(), random.Random(2))
    train = [s for s in a if s.kind == "train"]
    assert len(train) == 16 and all(s.seed < 9000 and s.policy_version == 2 for s in train)
    probes_a = [(s.opponent.id, s.seed, s.policy_slot, s.policy_civ) for s in a if s.kind == "probe"]
    probes_b = [(s.opponent.id, s.seed, s.policy_slot, s.policy_civ) for s in b if s.kind == "probe"]
    assert [p[0] for p in probes_a] == ["petra-1", "petra-2", "petra-3", "petra-4", "petra-5", "v1", "sonnet"]
    assert [p[1:] for p in probes_a if p[0] != "v1"] == [p[1:] for p in probes_b if p[0] != "v2"]
    first = run.plan_round(run.State(), [], args(frontier=""), random.Random(1))
    assert [s.opponent.id for s in first if s.kind == "probe"] == [f"petra-{n}" for n in range(1, 6)]


def test_a_round_writes_its_tasks_collects_its_games_and_renders_progress(tmp_path):
    specs = run.plan_round(run.State(), [], args(train_games=2, frontier=""), random.Random(1))
    bundle = run.write_bundle(specs, SETTINGS, tmp_path)
    assert {p.stem for p in (bundle / "tasks").glob("r00-*.json")} == {s.game_id for s in specs}
    first = specs[0]
    res = results(turns=[turn(0)], totals_at=[(0, 0, 0), (120_000, 50, 0)], final=(600, 100))
    if first.policy_slot == "2":
        res["seats"] = {"2": res["seats"]["1"], "1": res["seats"]["2"]}
        for row in res["timeline"]:
            row["players"] = {"2": row["players"]["1"], "1": row["players"]["2"]}
    (tmp_path / "games" / first.game_id / "inst-1").mkdir(parents=True)
    (tmp_path / "games" / first.game_id / "inst-1" / "results.json").write_text(json.dumps(res))
    records, decisions = run.collect(specs, tmp_path)
    assert records[0].outcome == "won" and records[0].margin == 500 and len(decisions) == 1
    assert all(r.status == "missing" for r in records[1:])
    rows = [json.loads(json.dumps(r.__dict__)) for r in records]
    page = run.progress(rows, run.rate(rows))
    assert "## Round 0 (policy v0)" in page and "| r00-t01 |" in page and "won | +500" in page


def fake_box(monkeypatch, trained, fail_train=None):
    """The round loop with every game, the trainer and vLLM faked: each game the policy wins by 500."""

    async def play(specs, bundle, round_dir, timeout_s):
        for s in specs:
            res = results(turns=[turn(0), turn(60_000)], totals_at=[(0, 0, 0), (120_000, 50, 0)], final=(600, 100))
            if s.policy_slot == "2":
                res["seats"] = {"2": res["seats"]["1"], "1": res["seats"]["2"]}
                for row in res["timeline"]:
                    row["players"] = {"2": row["players"]["1"], "1": row["players"]["2"]}
            out = round_dir / "games" / s.game_id / "inst"
            out.mkdir(parents=True)
            (out / "results.json").write_text(json.dumps(res))

    def train(args, home, round_, version, decisions_path, out_name=""):
        if fail_train is not None and len(trained) == fail_train:
            fail_train_once.append(1)
            raise RuntimeError("trainer died")
        trained.append((round_, version, decisions_path.parent.name))

    fail_train_once = []
    monkeypatch.setattr(run, "play", play)
    monkeypatch.setattr(run, "train", train)
    monkeypatch.setattr(run, "load_adapter", lambda args, version: None)
    monkeypatch.setattr(run, "ensure_adapters", lambda args, versions: None)
    monkeypatch.setenv("VLLM_API_KEY", "box-token")


def loop(tmp_path, *extra):
    argv = ["--home", str(tmp_path), "--train-games", "2", "--frontier", "", "--min-decisions", "1", *extra]
    return run.main(argv)


def test_the_loop_trains_on_the_baseline_first_and_never_replays_a_round_after_a_failed_training(tmp_path, monkeypatch):
    trained: list = []
    fake_box(monkeypatch, trained)
    assert loop(tmp_path, "--baseline") == 0
    state = run.State.load(tmp_path / "state.json")
    assert (state.round, state.version, state.pending_train) == (1, 0, True) and trained == []
    fake_box(monkeypatch, trained, fail_train=1)
    try:
        loop(tmp_path, "--rounds", "3")
    except RuntimeError as e:
        assert "trainer died" in str(e)
    state = run.State.load(tmp_path / "state.json")
    assert (state.round, state.version, state.pending_train) == (2, 1, True)  # round 1 played, its training failed
    fake_box(monkeypatch, trained)
    assert loop(tmp_path, "--rounds", "3") == 0
    state = run.State.load(tmp_path / "state.json")
    assert trained == [(0, 0, "r00"), (1, 1, "r01")] and (state.round, state.version) == (3, 2)
    assert state.snapshots == [2]
    rounds = [json.loads(line)["round"] for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert sorted(set(rounds)) == [0, 1, 2] and rounds.count(1) == 8  # 2 training, 5 Petra probes, v0: once
    assert "v1" in (tmp_path / "progress.md").read_text()
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["rounds"] == [0, 1, 2] and [r["version"] for r in rep["ratings"]] == [0, 1, 2]
    assert rep["state"]["round"] == 3 and all(r["policy"] > 1000 for r in rep["ratings"])  # it beats Petra every game
    assert {p["opponent"] for p in rep["probes"]} >= {"petra-1", "petra-5", "v0"} and rep["health"][0]["games"] == 7
    assert [g["round"] for g in rep["games"]] == [0, 2] and rep["games"][0]["points"][-1] == {
        "t_ms": 120_000, "policy": 50, "opponent": 0}  # fmt: skip
    assert sum(t["games"] for t in rep["training"] if t["round"] == 2) == 2


def test_on_a_one_gpu_box_vllm_steps_aside_for_the_trainer_and_comes_back_even_when_training_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: calls.append(cmd[:2] + cmd[-1:]))
    monkeypatch.setattr(run, "wait_healthy", lambda args, timeout_s: calls.append(["healthy"]))
    shared = argparse.Namespace(share_gpu=True, vllm_container="zeroad-vllm", vllm_wait_s=900)
    try:
        with run.gpu_for_training(shared):
            calls.append(["train"])
            raise RuntimeError("trainer died")
    except RuntimeError:
        pass
    assert calls == [["docker", "stop", "zeroad-vllm"], ["train"], ["docker", "start", "zeroad-vllm"], ["healthy"]]
    calls.clear()
    with run.gpu_for_training(argparse.Namespace(share_gpu=False)):
        calls.append(["train"])
    assert calls == [["train"]]


def test_runs_share_one_rating_scale_through_the_opponents_they_all_played():
    from agentenv_zeroad.pilot import combine

    strong = {"results": [[0, 0, "petra-3", 0.0]] * 3 + [[1, 1, "petra-3", 1.0]] * 6 + [[1, 1, "v0", 1.0]]}
    weak = {"results": [[0, 0, "petra-3", 0.0]] * 6 + [[1, 1, "petra-3", 0.5]] * 4}
    out = combine.combine({"a": strong, "b": weak})
    lines = {line["run"]: dict(line["points"]) for line in out["pooled"]["lines"]}
    assert out["pooled"]["board"]["petra-3"] == 1000 and set(out["pooled"]["board"]) == {"petra-3"}
    assert lines["a"][1] > lines["b"][1] > lines["b"][0] and lines["a"][1] > lines["a"][0]
    assert [r["id"] for r in out["runs"]] == ["a", "b"]


def test_a_game_whose_opponent_never_played_is_void_and_results_are_found_at_the_instance_path(tmp_path):
    frontier = Opponent("opus", "frontier", model="anthropic/claude-opus-5-5", trainable=False)
    specs = [spec(frontier, kind="probe"), spec(petra(2))]
    for i, s in enumerate(specs):
        specs[i] = GameSpec(f"r03-g{i}", 3, s.kind, 42, s.opponent, 2, "1", "rome", "pers")
    res = results(turns=[turn(0)], totals_at=[(0, 0, 0)], final=(900, 100))
    res["seats"]["2"] |= {"kind": "agent", "forfeited": True, "brain_calls": [{"error": "HTTPStatusError: 401\nmore"}]}
    nested = tmp_path / "games" / "r03-g0" / "@local" / "var" / "lib" / "bundle" / "r03-g0-ab12"
    nested.mkdir(parents=True)
    (nested / "zeroad-lb-1b0b-results.json").write_text(json.dumps(res))
    fine = tmp_path / "games" / "r03-g1" / "@local" / "x"
    fine.mkdir(parents=True)
    (fine / "zeroad-lb-2c1c-results.json").write_text(
        json.dumps(results(turns=[], totals_at=[(0, 0, 0)], final=(1, 2)))
    )
    records, _ = run.collect(specs, tmp_path)
    assert records[0].status == "void" and records[0].outcome is None and "401" in records[0].detail
    assert records[1].status == "finished" and records[1].outcome == "won"  # the helper's policy seat always won


def test_a_rounds_specs_are_rebuilt_from_its_task_files_and_recollect_replaces_only_that_rounds_records(tmp_path):
    specs = run.plan_round(run.State(round=0, version=0), [], args(train_games=2), random.Random(1))
    round_dir = tmp_path / "rounds" / "r00"
    run.write_bundle(specs, SETTINGS, round_dir)
    rebuilt = run.load_specs(round_dir)
    assert [(s.game_id, s.opponent.id, s.seed, s.policy_slot, s.kind) for s in rebuilt] == sorted(
        (s.game_id, s.opponent.id, s.seed, s.policy_slot, s.kind) for s in specs
    )
    (tmp_path / "games.jsonl").write_text(
        json.dumps({"round": 0, "game_id": "old", "kind": "train", "opponent": "petra-1", "status": "missing"}) + "\n"
        + json.dumps({"round": 1, "game_id": "keep", "kind": "train", "opponent": "petra-1", "status": "missing",
                      "policy_version": 1, "outcome": None, "margin": None, "calls": 0, "missed": 0, "rejected": 0,
                      "orders": 0, "latency_p50_s": None, "overruns": None}) + "\n"
    )  # fmt: skip
    (tmp_path / "state.json").write_text(json.dumps({"round": 2, "version": 1}))
    run.recollect(tmp_path, 0)
    rows = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert "old" not in {r["game_id"] for r in rows} and "keep" in {r["game_id"] for r in rows}
    assert sum(1 for r in rows if r["round"] == 0) == len(specs)


def test_the_best_objective_keeps_only_above_expectation_decisions_with_clean_orders(tmp_path):
    rows = [
        {"advantage": 1.2, "rejected_share": 0.0, "token_ids": [1]},
        {"advantage": 2.0, "rejected_share": 0.5, "token_ids": [2]},  # a rejected order: dropped
        {"advantage": 0.2, "rejected_share": 0.0, "token_ids": [3]},  # under the threshold
        {"advantage": -1.0, "rejected_share": 0.0, "token_ids": [4]},
    ]
    path = tmp_path / "decisions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = [json.loads(line) for line in run.best_decisions(path, 0.5).read_text().splitlines()]
    assert kept == [{"advantage": 1.0, "rejected_share": 0.0, "token_ids": [1]}]


def pool_args(pool, me, members="a,b,c", wait=0.0):
    return argparse.Namespace(pool=str(pool), pool_self=me, pool_members=members, pool_wait_s=wait, trainer_image="t")


def local_update(home, name, decisions):
    out = home / "adapters" / name
    out.mkdir(parents=True)
    (out / "adapter_config.json").write_text("{}")
    (out / "train.json").write_text(json.dumps({"decisions": decisions, "mean_kl": 0.01}))
    return out


def test_a_pool_round_averages_every_member_on_the_same_version_and_leaves_out_the_rest(tmp_path, monkeypatch):
    averaged = {}

    def fake_average(args, home, members, out):
        averaged[home.name] = [json.loads((m / "train.json").read_text())["decisions"] for m in members]
        out.mkdir(parents=True)

    monkeypatch.setattr(run, "average", fake_average)
    pool = tmp_path / "pool"
    homes = {m: tmp_path / m for m in "abc"}
    b = local_update(homes["b"], "local-v3", 200)
    run._pool_put(str(pool), "r04/b", b, {"member": "b", "from_version": 2, "decisions": 200})
    c = local_update(homes["c"], "local-v3", 80)
    run._pool_put(str(pool), "r04/c", c, {"member": "c", "from_version": 1, "decisions": 80})  # out of step
    a = local_update(homes["a"], "local-v3", 150)
    members = run.pool_round(pool_args(pool, "a"), homes["a"], 4, 2, a)
    assert members == ["a", "b"] and averaged["a"] == [150, 200]  # this box first; c trained from v1: left out
    assert (homes["a"] / "adapters" / "v3").is_dir() and (homes["a"] / "adapters" / "pool" / "r04" / "b").is_dir()
    assert json.loads((pool / "r04" / "a.json").read_text())["decisions"] == 150


def test_combined_ratings_treat_a_pooled_version_as_one_player_in_every_run():
    from agentenv_zeroad.pilot import combine

    a = {"results": [[0, 0, "petra-3", 0.0]] * 3 + [[1, 1, "petra-3", 1.0]] * 3}
    b = {"results": [[0, 0, "petra-3", 0.0]] * 3 + [[1, 1, "petra-3", 1.0]] * 3 + [[1, 1, "v0", 1.0]]}
    out = combine.combine({"a": a, "b": b}, shared=True)
    [line] = out["pooled"]["lines"]
    points = dict(line["points"])
    assert line["run"] == "pool" and points[1] > 1000 > points[0]
