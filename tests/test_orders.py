import pytest
from pydantic import ValidationError

from agentenv_zeroad.orders import (
    ORDER_TYPES,
    AdvancePhase,
    Army,
    Build,
    OrderBook,
    Role,
    SetEconomy,
    Train,
    call_text,
    parse_order,
    rejected,
)
from agentenv_zeroad.state import ROLES


def test_roles_in_the_vocabulary_are_the_snapshot_roles():
    assert set(Role.__args__) == set(ROLES)


def test_every_order_has_a_json_schema_with_its_kind_for_mcp_tools():
    for order_type in ORDER_TYPES:
        schema = order_type.model_json_schema()
        assert schema["type"] == "object" and "kind" in schema["properties"]
        assert schema.get("additionalProperties") is False


def test_economy_split_must_add_up_to_100():
    assert SetEconomy(food=50, wood=30, stone=10, metal=10).split() == {
        "food": 50,
        "wood": 30,
        "stone": 10,
        "metal": 10,
    }
    with pytest.raises(ValidationError, match="must be 100"):
        SetEconomy(food=50, wood=30, stone=10, metal=0)


def test_parse_order_picks_the_model_by_kind_and_rejects_unknown_fields():
    assert parse_order({"kind": "train", "role": "cavalry", "count": 3}) == Train(role="cavalry", count=3)
    assert isinstance(parse_order({"kind": "advance_phase"}), AdvancePhase)
    with pytest.raises(ValidationError):
        parse_order({"kind": "train", "role": "dragon", "count": 3})
    with pytest.raises(ValidationError):
        parse_order({"kind": "train", "role": "cavalry", "count": 0})
    with pytest.raises(ValidationError):
        parse_order({"kind": "say", "text": "hi", "volume": 11})


@pytest.mark.parametrize("where", ["base", "front", "near_wood", "near_metal", "toward_2", "toward_player_3"])
def test_build_accepts_every_where_form(where):
    assert Build(structure="house", where=where).where == where


@pytest.mark.parametrize("where", ["near_gold", "toward_", "behind", ""])
def test_build_rejects_other_where_values(where):
    with pytest.raises(ValidationError):
        Build(structure="house", where=where)


def test_army_targets_name_a_player_an_area_or_home():
    assert Army(action="attack", target="player_2").target_player == 2
    assert Army(action="raid", target="3").target_player == 3
    assert Army(action="defend").target_player is None and Army(action="defend").target == "home"
    with pytest.raises(ValidationError):
        Army(action="attack", target="the moon")
    with pytest.raises(ValidationError):
        Army(action="dance", target="home")


def test_results_carry_message_alternatives_and_a_suggested_call():
    result = rejected(
        "unknown_structure", "No such building.", ["house", "barracks"], call_text(Build(structure="house"))
    )
    assert not result.ok
    assert result.to_json() == {
        "status": "rejected",
        "message": "No such building.",
        "code": "unknown_structure",
        "alternatives": ["house", "barracks"],
        "suggest": "build(structure='house', where='base')",
    }


def test_order_book_summary_lists_standing_orders():
    book = OrderBook(1)
    assert sum(book.split.values()) == 100
    assert book.summary()["army"] == "gather home"
