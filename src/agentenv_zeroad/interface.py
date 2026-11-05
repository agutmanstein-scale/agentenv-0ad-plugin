"""The interface switch: which version of the seat-facing surface a seat plays (docs/post-training-v2.md, §1).

`v1` is the interface of the live stream and the 2026-11-04 pilot, byte for byte: its prompt, tool schemas, briefs and
order handling are what they were before the switch existed. `v2` is the frozen interface the post-training baseline
and run use: waves, a prompt with the real rules, order feedback in the memory, a forgiving `at`, a villager lever,
`hold()`, one `say()` per turn, numbered players and a richer brief.

The default is `v1` everywhere, so nothing changes unless a seat asks. The seat's environment side reads it from the
slot's `interface` setting (`SeatPlan.interface`: the brief, the executor and the order checks) and its brain side from
`ZEROAD_INTERFACE` (`CommanderSettings.interface`: the prompt, the tool list and the memory). Both must name the same
version; the pilot's task builder sets both from one `PilotSettings.interface`.

The env's MCP tool list is the same for every seat: it always carries `hold` and the v2 parameters (`min_size`,
`regroup_below`, `villagers`), and a v1 seat that calls them gets a `not_in_v1` rejection. Only the commander, the one
brain in this repo, trims the list to its seat's interface (`SeatLink.tools`), so a v1 seat's model sees exactly
v1's tools. A different MCP client on a v1 seat would be offered the v2 ones.
"""

from __future__ import annotations

from typing import Literal

Interface = Literal["v1", "v2"]
INTERFACES: tuple[Interface, ...] = ("v1", "v2")
DEFAULT_INTERFACE: Interface = "v1"
ENV_VAR = "ZEROAD_INTERFACE"


def parse_interface(value: str | None) -> Interface:
    """`v1` or `v2` (an unset or empty value is the default); anything else is an error, never a silent v1."""
    if not value:
        return DEFAULT_INTERFACE
    if value not in INTERFACES:
        raise ValueError(f"{ENV_VAR} must be one of {', '.join(INTERFACES)}, not {value!r}")
    return value  # type: ignore[return-value]
