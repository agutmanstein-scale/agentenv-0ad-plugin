"""Which SFT epoch checkpoint to keep: held-out win rate with an entropy floor, never the SFT loss."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agentenv_zeroad.train import report as R
from agentenv_zeroad.train import stats
from tests.train_fakes import results, turn

C = stats.SftCandidate


def test_the_best_panel_win_rate_wins_among_checkpoints_that_kept_their_entropy():
    pick = stats.select_sft_checkpoint([C("epoch1", 0.30, 40, 0.92), C("epoch2", 0.40, 40, 0.81)])
    assert pick["chosen"] == "epoch2" and [c["rejected"] for c in pick["candidates"]] == [None, None]


def test_a_checkpoint_whose_entropy_fell_below_the_floor_is_rejected_however_well_it_plays():
    pick = stats.select_sft_checkpoint([C("epoch1", 0.30, 40, 0.92), C("epoch2", 0.55, 40, 0.55)])
    assert pick["chosen"] == "epoch1"
    assert "0.55 of base is under 0.7" in pick["candidates"][1]["rejected"]


def test_the_floor_is_configurable_and_the_boundary_passes():
    both = [C("epoch1", 0.30, 40, 0.7), C("epoch2", 0.50, 40, 0.6)]
    assert stats.select_sft_checkpoint(both)["chosen"] == "epoch1"  # exactly 0.7 is not below it
    assert stats.select_sft_checkpoint(both, min_entropy_ratio=0.5)["chosen"] == "epoch2"
    assert stats.select_sft_checkpoint(both, min_entropy_ratio=0.5)["min_entropy_ratio"] == 0.5
    assert stats.MIN_ENTROPY_RATIO == 0.7


def test_a_tie_goes_to_the_earlier_less_trained_checkpoint():
    assert stats.select_sft_checkpoint([C("epoch1", 0.4, 40, 0.9), C("epoch2", 0.4, 40, 0.9)])["chosen"] == "epoch1"


def test_nothing_chosen_means_no_sft_start_and_unplayed_checkpoints_are_never_chosen():
    pick = stats.select_sft_checkpoint(
        [C("epoch1", 0.3, 40, 0.4), C("epoch2", None, 0, 0.9), C("epoch3", 0.6, 40, None)]
    )
    assert pick["chosen"] is None
    assert [c["rejected"] for c in pick["candidates"]][1:] == ["not played on the panel", "no entropy measured"]


def test_the_rule_has_no_loss_input_to_choose_by():
    assert {f.name for f in dataclasses.fields(C)} == {"name", "win_rate", "games", "entropy_ratio"}
    assert "never the SFT loss" in stats.select_sft_checkpoint([])["rule"]


# ---- from the folders -------------------------------------------------------------------------------------------
def epoch_manifest(epochs: Path, n: int, version: int, ratio: float) -> None:
    folder = epochs / f"epoch{n}"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"epoch": n, "version": version, "entropy_ratio": ratio}))


def panel_games(panels: Path, name: str, outcomes: list[str]) -> None:
    for i, outcome in enumerate(outcomes):
        folder = panels / name / f"g{i}"
        folder.mkdir(parents=True)
        r = results(
            f"{name}-g{i}", outcomes=(outcome, "lost" if outcome == "won" else "won"), model_1="local/policy-v2",
            turns={"1": [turn(0, model="local/policy-v2")]},
        )  # fmt: skip
        (folder / "results.json").write_text(json.dumps(r))


def test_the_report_reads_each_epochs_panel_and_manifest_and_rejects_the_collapsed_one(tmp_path):
    epochs, panels = tmp_path / "epochs", tmp_path / "panels"
    epoch_manifest(epochs, 1, 3, 0.9)
    epoch_manifest(epochs, 2, 6, 0.5)
    panel_games(panels, "sft-epoch1", ["won", "lost", "lost", "lost"])
    panel_games(panels, "sft-epoch2", ["won", "won", "won", "lost"])
    chosen = R.select_sft(epochs, panels, None, "sft-epoch", 0.7)
    assert chosen["chosen"] == "epoch1" and chosen["versions"] == {"epoch1": 3, "epoch2": 6}
    rows = {c["name"]: c for c in chosen["candidates"]}
    assert rows["epoch1"]["win_rate"] == 0.25 and rows["epoch2"]["win_rate"] == 0.75 and rows["epoch2"]["rejected"]


def test_the_report_has_the_selection_only_when_given_the_epochs_and_orders_epochs_numerically(tmp_path):
    epochs, panels, games = tmp_path / "epochs", tmp_path / "panels", tmp_path / "games"
    games.mkdir()
    for n in (2, 10):
        epoch_manifest(epochs, n, n, 0.9)
        panel_games(panels, f"sft-epoch{n}", ["won"])
    built = R.build(games, panels=panels, sft_epochs=epochs, now=1.0, bootstrap=0)
    assert [c["name"] for c in built["sft_selection"]["candidates"]] == ["epoch2", "epoch10"]
    assert built["sft_selection"]["chosen"] == "epoch2"
    assert R.build(games, panels=panels, now=1.0, bootstrap=0)["sft_selection"] is None
