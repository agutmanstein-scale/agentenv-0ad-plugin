from agentenv_zeroad.grading import (
    BRAIN_TIMEOUT_S,
    BrainCall,
    conquest_outcomes,
    forfeited,
    moves,
    seat_score,
    time_cap_outcomes,
)


def test_a_call_that_errors_or_runs_past_120_s_is_missed_and_a_slow_answer_is_not():
    assert BrainCall(seconds=3, error="HTTP 500").missed
    assert BrainCall(seconds=BRAIN_TIMEOUT_S + 0.1).missed
    assert not BrainCall(seconds=BRAIN_TIMEOUT_S).missed
    assert not BrainCall(seconds=90).missed


def test_more_than_half_missed_forfeits_and_exactly_half_does_not():
    ok, late, broken = BrainCall(10), BrainCall(200), BrainCall(1, error="timeout")
    assert moves([ok, late, broken]) == (1, 2)
    assert forfeited([ok, late, broken])
    assert not forfeited([ok, ok, late, broken])
    assert not forfeited([])


def test_slow_but_answering_models_are_never_zeroed():
    slow = [BrainCall(110)] * 10
    assert not forfeited(slow)
    assert seat_score("won", slow, graded_by_calls=True) == 1.0


def test_seat_score_is_win_one_draw_half_loss_zero_and_zero_when_forfeited_or_silent():
    calls = [BrainCall(5)]
    assert [seat_score(o, calls, graded_by_calls=True) for o in ("won", "drawn", "lost")] == [1.0, 0.5, 0.0]
    assert seat_score("won", [BrainCall(1, error="x")] * 3, graded_by_calls=True) == 0.0
    assert seat_score("won", [], graded_by_calls=True) == 0.0
    assert seat_score("won", [], graded_by_calls=False) == 1.0


def test_at_the_time_cap_the_highest_score_wins_and_everyone_else_loses():
    assert time_cap_outcomes({"1": 300, "2": 410, "3": 50}) == {"1": "lost", "2": "won", "3": "lost"}


def test_at_the_time_cap_a_tie_on_top_is_a_draw_between_the_leaders():
    assert time_cap_outcomes({"1": 400, "2": 400, "3": 10}) == {"1": "drawn", "2": "drawn", "3": "lost"}
    assert time_cap_outcomes({"1": 400, "2": 395}, draw_margin=5) == {"1": "drawn", "2": "drawn"}
    assert time_cap_outcomes({"1": 400, "2": 394}, draw_margin=5) == {"1": "won", "2": "lost"}


def test_conquest_follows_the_engines_player_states():
    assert conquest_outcomes({"1": "active", "2": "active"}) is None
    assert conquest_outcomes({"1": "won", "2": "defeated"}) == {"1": "won", "2": "lost"}
    assert conquest_outcomes({"1": "active", "2": "defeated", "3": "defeated"}) == {
        "1": "won",
        "2": "lost",
        "3": "lost",
    }
    assert conquest_outcomes({"1": "active", "2": "active", "3": "defeated"}) is None
    assert conquest_outcomes({"1": "defeated", "2": "defeated"}) == {"1": "drawn", "2": "drawn"}
