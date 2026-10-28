import asyncio

import pytest

from agentenv_zeroad.engine import FakeEngine
from agentenv_zeroad.roles import RoleCatalog, parse_template, role_of
from agentenv_zeroad.state import ROLES
from tests.executor_fixtures import rome_table, rome_templates


@pytest.mark.parametrize(
    ("classes", "role"),
    [
        ({"Unit", "Infantry", "Melee", "Soldier"}, "infantry_melee"),
        ({"Unit", "Infantry", "Ranged"}, "infantry_ranged"),
        ({"Unit", "Cavalry", "Ranged"}, "cavalry"),
        ({"Unit", "Champion", "Infantry"}, "champion"),
        ({"Unit", "Siege"}, "siege"),
        ({"Unit", "Hero", "Cavalry"}, "other"),
        ({"Unit", "FemaleCitizen"}, "villager"),
        ({"Unit", "Civilian", "Worker"}, "villager"),
        ({"Unit", "Ship"}, "other"),
    ],
)
def test_role_rules_match_the_snapshot_query(classes, role):
    assert role_of(classes) == role


def test_parse_template_expands_placeholders_and_reads_cost_requirements_and_lists():
    xml = (
        '<Identity><Classes datatype="tokens">Structure</Classes><VisibleClasses datatype="tokens">Village House'
        '</VisibleClasses><Requirements><Techs datatype="tokens">phase_town</Techs></Requirements></Identity>'
        "<Cost><Population>0</Population><Resources><food>0</food><wood>150</wood></Resources></Cost>"
        '<Trainer><Entities datatype="tokens">units/{native}/support_civilian units/{civ}/cavalry_b</Entities>'
        "</Trainer>"
    )
    tpl = parse_template("structures/gaul/house", xml, "gaul")
    assert tpl.key == "house" and tpl.kind == "House"
    assert tpl.cost == {"wood": 150} and tpl.requires == ("phase_town",)
    assert tpl.trains == ("units/gaul/support_civilian", "units/gaul/cavalry_b")


def test_rome_table_from_recorded_engine_templates_covers_every_role_and_core_building():
    table = rome_table()
    assert set(table.roles()) == set(ROLES)
    assert table.villager is not None and table.villager.role == "villager"
    for key in ("civil_centre", "house", "storehouse", "farmstead", "field", "barracks", "stable"):
        assert table.structure(key) is not None, key
    assert table.structure("house").kind == "House" and table.structure("storehouse").drop
    assert "barracks" in table.trained_at("infantry_melee")
    assert all(name.startswith("units/rome/") for name in table.templates_for("infantry_ranged"))


def test_catalog_caches_each_civ_and_skips_names_the_engine_does_not_know():
    class Counting(FakeEngine):
        calls = 0

        async def templates(self, names):
            Counting.calls += 1
            return await super().templates(names)

    engine = Counting(templates=rome_templates())
    catalog = RoleCatalog(engine)

    async def go():
        first = await catalog.table("rome")
        calls = Counting.calls
        second = await catalog.table("rome")
        return first, second, calls

    first, second, calls = asyncio.run(go())
    assert first is second and Counting.calls == calls == 3
    # The Builder list names structures other civs have (e.g. a crannog); they are absent here, not guessed.
    assert "crannog" not in first.structures and "army_camp" in first.structures


def test_a_civ_without_a_female_citizen_gets_its_buildings_from_the_civic_centres_villager():
    # Germans (0.28) have no `support_female_citizen`; their civic centre trains `support_civilian`, the builder.
    templates = {k: v for k, v in rome_templates().items() if k != "units/rome/support_female_citizen"}
    table = asyncio.run(RoleCatalog(FakeEngine(templates=templates)).table("rome"))
    for key in ("house", "storehouse", "farmstead", "field", "barracks"):
        assert table.structure(key) is not None, key
    assert table.structure_of_kind("House") is not None
    assert table.villager is not None and table.villager.name == "units/rome/support_civilian"
    assert set(table.roles()) == set(ROLES)


def test_the_catalog_asks_the_engine_to_load_only_templates_that_exist_so_setup_logs_no_errors():
    # A rendered match shows every "Failed to load entity template" on screen; the civ's lists name templates other
    # civs have (e.g. a crannog), so the catalog checks existence first, which loads nothing.
    engine = FakeEngine(templates=rome_templates())
    table = asyncio.run(RoleCatalog(engine).table("rome"))
    asked = [name for batch in engine.template_requests for name in batch]
    assert asked and all(name in rome_templates() for name in asked)
    assert "structures/rome/crannog" not in asked and "army_camp" in table.structures


def test_the_fixture_recorder_still_drives_the_role_catalog_and_keeps_every_template_it_loaded():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "record_executor_fixtures.py"
    spec = importlib.util.spec_from_file_location("record_executor_fixtures", script)
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    recording = recorder.Recording(FakeEngine(templates=rome_templates()))
    table = asyncio.run(RoleCatalog(recording).table("rome"))
    assert "barracks" in table.structures
    assert "units/rome/support_female_citizen" in recording.seen
