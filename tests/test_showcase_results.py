import copy
import json
from pathlib import Path
from typing import Any

from agentenv_zeroad.showcase.results import GameRow, Session, classify, verify_lockstep

FIXTURE = Path(__file__).parent / "fixtures" / "showcase_results_frontier.json"


def frontier() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def results(
    *,
    policy: str = "won",
    ended_by: str = "conquest",
    failed: dict[str, int] | None = None,
    decisions: dict[str, int] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A results dict with the fields `classify` reads: policy in slot 1, an opponent (Opus) in slot 2."""
    decisions = decisions or {"1": 20, "2": 20}
    failed = failed or {}
    other = {"won": "lost", "lost": "won", "drawn": "drawn"}[policy]
    return {
        "status": "finished",
        "ended_by": ended_by,
        "seats": {"1": {"outcome": policy}, "2": {"outcome": other}},
        "lockstep": {
            "latency_ms": 5000,
            "decisions": [
                {"slot": slot, "frozen_ms": i * 25000, "due_ms": i * 25000 + 5000, "applied_ms": i * 25000 + 5000}
                for slot, n in decisions.items()
                for i in range(n)
            ],
        },
        "sampler": {slot: {"calls": decisions[slot], "failed_calls": failed.get(slot, 0)} for slot in decisions},
        **extra,
    }


def classified(res: dict[str, Any], opponent: str = "Opus"):
    return classify(res, index=3, opponent=opponent, policy_slot="1")


def test_win_by_conquest() -> None:
    row = classified(results())
    assert (row.result, row.how, row.reason) == ("win", "conquest", None)
    assert (row.index, row.opponent, row.policy_slot) == (3, "Opus", "1")


def test_win_by_net_worth_at_cap() -> None:
    row = classified(results(ended_by="time_cap"))
    assert (row.result, row.how) == ("win", "net_worth")


def test_loss_is_a_loss() -> None:
    row = classified(results(policy="lost", ended_by="time_cap"))
    assert (row.result, row.how) == ("loss", "net_worth")


def test_exact_tie_is_a_draw() -> None:
    row = classified(results(policy="drawn", ended_by="time_cap"))
    assert (row.result, row.how) == ("draw", "net_worth")


def test_engine_failure_is_no_result() -> None:
    row = classified(results(failed=None, status="failed", detail="the engine stopped stepping", ended_by="failed"))
    assert row.result == "no_result"
    assert row.how is None
    assert row.reason == "engine failed"
    assert row.blame == "engine"


def test_failed_key_is_no_result() -> None:
    res = results()
    res["failed"] = "the box went away"
    row = classified(res)
    assert (row.result, row.reason) == ("no_result", "engine failed")


def test_engine_failure_reason_carries_no_url() -> None:
    detail = "ConnectError: http://10.0.0.5:8000/v1 ... s3://some-bucket/x"
    row = classified(results(status="failed", detail=detail, ended_by="failed"))
    assert row.reason == "engine failed (ConnectError)"
    assert "://" not in row.reason and "some-bucket" not in row.reason


def test_ten_percent_failed_calls_is_no_result_either_side() -> None:
    assert classified(results(failed={"2": 3})).result == "no_result"
    assert classified(results(failed={"2": 2})).result == "win"
    assert classified(results(failed={"1": 3})).result == "no_result"
    assert classified(results(failed={"1": 2})).result == "win"


def test_no_result_reason_names_the_model_and_the_count() -> None:
    row = classified(results(failed={"2": 4}), opponent="Astra")
    assert row.reason == "Astra, 4 failed calls"
    assert row.blame == "opponent"
    assert row.failed == {"policy": 0, "opponent": 4}
    assert row.decisions == {"policy": 20, "opponent": 20}
    mine = classified(results(failed={"1": 4}), opponent="Astra")
    assert mine.reason == "Qwen, 4 failed calls"
    assert mine.blame == "policy"


def test_vllm_errors_count_as_failed_calls() -> None:
    row = classified(results(failed={"1": 5}, decisions={"1": 30, "2": 30}))
    assert row.result == "no_result"
    assert row.blame == "policy"


def test_timeouts_are_not_failed_calls() -> None:
    res = results(failed={}, decisions={"1": 20, "2": 20})
    for d in [d for d in res["lockstep"]["decisions"] if d["slot"] == "2"][:4]:
        d.update(applied_ms=None, missed=True)
    missed = [d for d in res["lockstep"]["decisions"] if d.get("missed")]
    assert missed
    assert classified(res).result == "win"


def test_petra_has_no_calls() -> None:
    res = results(decisions={"1": 20})
    res["seats"]["2"]["kind"] = "petra"
    row = classified(res, opponent="Petra-3")
    assert row.result == "win"
    assert row.failed == {"policy": 0, "opponent": 0}
    assert row.decisions == {"policy": 20, "opponent": 0}


def test_ended_early_is_no_result_with_reason() -> None:
    ended = {"slot": "2", "failures": 5}
    res = results(failed={"2": 5}, decisions={"1": 60, "2": 60}, showcase={"ended_early": ended})
    row = classified(res, opponent="Opus")
    assert row.result == "no_result"
    assert row.reason == "Opus, 5 failed calls in a row"
    assert row.blame == "opponent"


def test_ended_early_blames_the_slot_that_failed() -> None:
    res = results(
        failed={"1": 9, "2": 5},
        decisions={"1": 200, "2": 200},
        showcase={"ended_early": {"slot": "2", "failures": 5}},
    )
    assert classified(res).blame == "opponent"
    res["showcase"]["ended_early"]["slot"] = "1"
    assert classified(res).blame == "policy"


def test_verify_lockstep_passes_a_13_3_game() -> None:
    assert verify_lockstep(frontier()) == []


def test_fixture_classifies_as_the_recorded_game() -> None:
    row = classify(frontier(), index=0, opponent="Opus", policy_slot="1")
    assert (row.result, row.how) == ("win", "net_worth")


def test_verify_lockstep_flags_an_order_off_t_plus_5() -> None:
    res = frontier()
    res["lockstep"]["decisions"][4]["applied_ms"] += 40
    problems = verify_lockstep(res)
    assert len(problems) == 1
    assert "5000" in problems[0]


def test_verify_lockstep_flags_the_spec() -> None:
    res = frontier()
    spec = copy.deepcopy(res["spec"])
    spec["latency_ms"] = 2000
    spec["time_cap_minutes"] = 20
    spec["score_rule"] = "official"
    spec["seats"][0]["interface"] = "v1"
    spec["seats"][0]["sampling"] = {"temperature": 0.7}
    res["spec"] = spec
    problems = " ".join(verify_lockstep(res))
    for needle in ("latency", "30", "net_worth", "v2", "sampling"):
        assert needle in problems


def row(
    index: int, opponent: str, result: str = "win", *, blame: str | None = None, seed: int | None = None
) -> GameRow:
    return GameRow(
        index=index,
        game_id=f"showcase-{index}",
        opponent=opponent,
        seed=seed if seed is not None else 900400 + index,
        policy_slot="1",
        result=result,  # type: ignore[arg-type]
        how="conquest" if result in ("win", "loss", "draw") else None,
        reason=None if result != "no_result" else f"{opponent}, 9 failed calls",
        failed={"policy": 0, "opponent": 9 if blame == "opponent" else 0},
        decisions={"policy": 20, "opponent": 20},
        played_at=1000.0 + index,
        problems=[],
        blame=blame,
    )


def test_table_in_rotation_order_counts_no_results(tmp_path: Path) -> None:
    session = Session(tmp_path / "session.json")
    for r in (
        row(0, "Opus", "win"),
        row(1, "Astra", "loss"),
        row(2, "Petra-3", "draw"),
        row(3, "Opus", "no_result", blame="opponent"),
        row(4, "Astra", "win"),
    ):
        session.add(r)
    assert session.table() == [
        {"opponent": "Opus", "w": 1, "l": 0, "d": 0, "nr": 1},
        {"opponent": "Astra", "w": 1, "l": 1, "d": 0, "nr": 0},
        {"opponent": "Petra-3", "w": 0, "l": 0, "d": 1, "nr": 0},
    ]


def test_two_no_results_in_a_row_drop_an_opponent(tmp_path: Path) -> None:
    session = Session(tmp_path / "session.json")
    session.add(row(0, "Opus", "no_result", blame="opponent"))
    assert session.dropped() == set()
    session.add(row(1, "Astra", "win"))
    session.add(row(2, "Opus", "no_result", blame="opponent"))
    assert session.dropped() == {"Opus"}
    session.add(row(3, "Opus", "win"))
    assert session.dropped() == set()


def test_no_results_from_the_policy_do_not_drop_the_opponent(tmp_path: Path) -> None:
    session = Session(tmp_path / "session.json")
    session.add(row(0, "Opus", "no_result", blame="policy"))
    session.add(row(1, "Opus", "no_result", blame="engine"))
    session.add(row(2, "Opus", "no_result", blame="policy"))
    assert session.dropped() == set()


def test_session_round_trips_and_next_index_follows_the_last_row(tmp_path: Path) -> None:
    path = tmp_path / "out" / "session.json"
    session = Session(path)
    assert session.next_index() == 0
    session.add(row(0, "Opus"))
    session.add(row(1, "Astra", "no_result", blame="opponent"))
    session.ended = "budget"
    session.ended_why = "the hour ran out"
    session.save()

    data = json.loads(path.read_text())
    assert [r["seed"] for r in data["rows"]] == [900400, 900401]
    assert data["ended"] == "budget"
    assert not path.with_suffix(".tmp").exists()

    again = Session(path)
    again.load()
    assert again.rows == session.rows
    assert (again.ended, again.ended_why) == ("budget", "the hour ran out")
    assert again.next_index() == 2
    assert again.seeds() == {900400, 900401}


def test_load_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    session = Session(tmp_path / "none.json")
    session.load()
    assert session.rows == []
    assert session.ended is None


def test_table_uses_rotation_order_and_lists_unplayed_opponents(tmp_path: Path) -> None:
    session = Session(tmp_path / "session.json", rotation=("opus", "astra", "petra-3"))
    session.add(row(0, "astra", "win"))
    assert session.table() == [
        {"opponent": "opus", "w": 0, "l": 0, "d": 0, "nr": 0},
        {"opponent": "astra", "w": 1, "l": 0, "d": 0, "nr": 0},
        {"opponent": "petra-3", "w": 0, "l": 0, "d": 0, "nr": 0},
    ]
