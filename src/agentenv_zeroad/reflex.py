"""The Luna reflex layer (Setup A, spec §4.6): quick tactical choices inside the strategist's standing orders.

A seat whose slot sets `reflex: luna` gets one `Reflex`. Every `interval_s` (1 s, spike results) it looks at the
seat's latest snapshot and order book and opens a question for each tactical moment that is open now. The questions
are the ones where the answer matters and can differ from the standing orders (U5's re-run showed that asking
whether to keep an order nobody was fighting over gave 0 orders in 8 minutes):
- `threat`: enemy soldiers are at the base. Under an attack or raid order: press on, bring the army home to defend,
  or fall back. Under any other order (not already defending at home): defend at the base, or ignore them and keep
  working (soldiers fight back where they stand);
- `regroup`: the army still follows a departure the reflex chose earlier and the danger has passed: resume the
  strategist's order, or stay a little longer;
- `engage`: our soldiers are in contact with enemy soldiers away from the base. Under an attack or raid order: fight
  their soldiers, push past them to the buildings, or fall back home; under any other order: hold or fall back;
- `target`: an attack or raid is out and no fight is on: hit the enemy's visible soldiers, its workers or its
  buildings first (the posture's `focus`, which the army driver turns into attack-walks);
- `counter`: during a fight (contact or enemies at the base), a production building's queue is free: which of the
  soldier roles the strategist has ordered to top up (by up to 3, to at most 6 queued) to counter what is in sight;
- `exhausted`: a resource the economy split works has just run out near the base (it had supply before; food doesn't
  run out while the seat has fields): which of the split's other resources its gatherers move to, or keep the split.
Luna may always keep the strategist's order, departs from it only as above, and never starts a war: no question
offers an attack the strategist didn't order. A departure is temporary: the strategist's order stays the standing
one, `regroup` offers it back, and a new strategist order replaces both. After an answer that changed something, the
same question rests for a few seconds, so a fight doesn't flip the army every second.

All open questions go in **one** Decisions request. The answers become executor orders through the seat's own book
(`Army`, `SetEconomy`, `Train`), so they go through the same checks as a strategist's orders; `target` and the
fight half of `engage` set the army's `focus`. Every answer that changed something is counted by kind and reported
once to the seat's action record (`on_action`: its orders as tool calls, their status, and a line for the overlay,
"pulled the army home to defend the base"), which results.json, the brief footer and the overlay read.

It never blocks anything: the reflex runs as its own task beside the clock and the seat loop; a request that is
still in flight when the next interval comes skips that interval (one in flight), a request past its timeout or one
that fails is dropped, and a seat without credentials, or whose spend guard has stopped, asks nothing. While a
request with a threat, engage or exhausted question is in flight the seat is *waiting* (spec §5), reported to the
seat's metrics; the other questions change a plan the executor is already carrying out, so they are no wait.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentenv_zeroad.brief import json_brief
from agentenv_zeroad.decisions import Answer, Decision, Question, choice
from agentenv_zeroad.metrics import SeatMetrics, percentile
from agentenv_zeroad.orders import Army, OrderBook, SetEconomy, Train, _Order, call_text
from agentenv_zeroad.spend import SpendCapReached
from agentenv_zeroad.state import RESOURCES, Event, PlayerSnapshot, Snapshot, _base_ccs

log = logging.getLogger(__name__)

INTERVAL_S = 1.0
WAITED_ON = ("threat", "engage", "exhausted")  # decisions `open_decisions` counts open; the rest adjust a plan
IDLE_ARMY_MIN = 3
CONTACT_MIN = 2  # enemy soldiers within reach of ours that make a fight
COUNTER_BATCH = 3
COUNTER_TOP = 2 * COUNTER_BATCH  # the counter question tops an ordered role up to at most this many queued
COUNTER_ROLES = ("infantry_melee", "infantry_ranged", "cavalry")
OFFENSIVE = ("attack", "raid")
QUIET_MS = {"threat": 8_000, "regroup": 8_000, "engage": 8_000, "target": 15_000, "counter": 10_000}
SYSTEM = (
    "You are the reflexes of an empire in a live 0 A.D. match. A strategist sets the standing orders every half "
    "minute; you see the battle every second. Keep the plan when it still fits, and change course when the moment "
    "calls for it: an attack on the base, a fight going badly, a better target in sight, the wrong soldiers in "
    "production. Viewers watch your calls live."
)


class Decider(Protocol):
    async def decide(self, context: str, questions: list[Question]) -> Decision: ...


@dataclass(frozen=True)
class Focus:
    """Set the army's attack focus (`ArmyPosture.focus`)."""

    focus: str


Act = _Order | Focus


@dataclass
class Open:
    """One open question, how each answer becomes orders (or a focus) for the seat, and the overlay line for each
    answer that changes something."""

    question: Question
    apply: Callable[[str], list[Act] | None]
    texts: dict[str, str] = field(default_factory=dict)
    key: str | None = None


def _army(me: PlayerSnapshot) -> int:
    return sum(n for role, n in me.units.items() if role not in ("villager", "other"))


def _enemy_soldiers(me: PlayerSnapshot) -> dict[str, int]:
    out: Counter[str] = Counter()
    for view in me.enemies.values():
        for role, n in view.units.items():
            if role not in ("villager", "other"):
                out[role] += n
    return dict(out)


def contact(me: PlayerSnapshot) -> int:
    return sum(v.contact for v in me.enemies.values())


def _restore(book: OrderBook, standing: tuple[str, str], current: tuple[str, str]) -> Callable[[Army], list[Act]]:
    """An army answer's orders: none when the strategist changed the army meanwhile or nothing would change."""

    def go(order: Army) -> list[Act]:
        if (book.army.action, book.army.target) != current or (order.action, order.target) == current:
            return []
        return [order]

    return go


def threat_question(me: PlayerSnapshot, book: OrderBook, standing: tuple[str, str]) -> Open | None:
    if me.enemies_near_base <= 0:
        return None
    action, target = standing
    current = (book.army.action, book.army.target)
    go = _restore(book, standing, current)
    keep = Army(action=action, target=target)
    home = Army(action="defend", target="home")
    if action in OFFENSIVE:
        options = {
            "press_on": (f"keep the army on its {action} on {target}; the base copes on its own", keep),
            "defend": ("bring the army home and defend the base", home),
            "fall_back": ("call off the attack and retreat home", Army(action="retreat", target="home")),
        }
        texts = {
            "press_on": f"kept the {action} on {target} going despite the raid at home",
            "defend": "pulled the army home to defend the base",
            "fall_back": "called off the attack and fell back home",
        }
    elif current == ("defend", "home"):
        return None
    else:
        options = {
            "defend": ("soldiers drop their work and fight the raiders at the base", home),
            "ignore": (f"keep to the standing order ({action} {target}); soldiers fight back where they stand", keep),
        }
        texts = {"defend": "sent the soldiers to defend the base", "ignore": f"went back to {action} {target}"}
    seen = ", ".join(f"{n} {r}" for r, n in _enemy_soldiers(me).items()) or "unseen"
    text = (
        f"{me.enemies_near_base} enemy soldiers are at our base (in sight: {seen}). We have {_army(me)} soldiers; "
        f"standing order: {action} {target}"
    ) + (f"; for now they {current[0]} {current[1]}." if current != standing else ".")

    def apply(value: str) -> list[Act]:
        picked = options.get(value)
        return go(picked[1]) if picked else []

    return Open(choice("threat", text, {v: d for v, (d, _) in options.items()}), apply, texts)


def regroup_question(me: PlayerSnapshot, book: OrderBook, standing: tuple[str, str]) -> Open | None:
    """The army still follows the reflex's own departure, and the danger that caused it has passed."""
    current = (book.army.action, book.army.target)
    if current == standing or me.enemies_near_base > 0 or contact(me) >= CONTACT_MIN:
        return None
    action, target = standing
    go = _restore(book, standing, current)
    options = {
        "resume": f"go back to the strategist's order: {action} {target}",
        "stay": f"stay on {current[0]} {current[1]} a little longer",
    }
    text = (
        f"The danger has passed. Our {_army(me)} soldiers {current[0]} {current[1]}; "
        f"the strategist wants {action} {target}."
    )

    def apply(value: str) -> list[Act]:
        return go(Army(action=action, target=target)) if value == "resume" else []

    return Open(choice("regroup", text, options), apply, {"resume": f"sent the army back to {action} {target}"})


def engage_question(me: PlayerSnapshot, book: OrderBook) -> Open | None:
    """Our soldiers meet enemy soldiers away from the base (the base has its own question). Fight or push is offered
    only under an attack on a player: the army driver applies a focus to nothing else."""
    n = contact(me)
    if n < CONTACT_MIN or me.enemies_near_base > 0:
        return None
    current = (book.army.action, book.army.target)
    seq = book.army.seq
    go = _restore(book, current, current)
    back = Army(action="retreat", target="home")
    seen = ", ".join(f"{k} {r}" for r, k in _enemy_soldiers(me).items()) or "unseen"
    text = (
        f"{n} enemy soldiers are fighting ours ({_army(me)} soldiers; in sight: {seen}); "
        f"order {current[0]} {current[1]}."
    )
    if current[0] in OFFENSIVE and Army(action="attack", target=current[1]).target_player is not None:
        options = {
            "fight": "turn on their soldiers and beat them first",
            "push": "push past their soldiers to the buildings",
            "fall_back": "the fight is lost: retreat home",
        }
        texts = {
            "fight": "turned the army on the enemy soldiers",
            "push": "pushed past their army to the buildings",
            "fall_back": "pulled the army out of a losing fight",
        }
        focus = {"fight": "army", "push": "buildings"}

        def apply(value: str) -> list[Act]:
            if value == "fall_back":
                return go(back)
            want = focus.get(value)
            if want is None or book.army.seq != seq or book.army.focus == want:
                return []
            return [Focus(want)]

    else:
        options = {"hold": "stand and fight where they are", "fall_back": "retreat home out of the fight"}
        texts = {"fall_back": "pulled the soldiers back home out of the fight"}

        def apply(value: str) -> list[Act]:
            return go(back) if value == "fall_back" else []

    return Open(choice("engage", text, options), apply, texts)


def target_question(me: PlayerSnapshot, book: OrderBook) -> Open | None:
    """An attack or raid is out, nothing is fighting it yet, and something of the target is in sight."""
    if book.army.action not in OFFENSIVE or contact(me) >= CONTACT_MIN:
        return None
    q = Army(action="attack", target=book.army.target).target_player
    enemy = me.enemies.get(q) if q is not None else None
    if enemy is None:
        return None
    options: dict[str, str] = {}
    if enemy.soldiers_at is not None:
        options["army"] = f"their soldiers ({enemy.soldiers_at[2]} in sight)"
    if enemy.workers_at is not None:
        options["workers"] = f"their gatherers ({enemy.workers_at[2]} in sight)"
    if enemy.buildings:
        options["buildings"] = f"their buildings ({len(enemy.buildings)} in sight)"
    if len(options) < 2:
        return None
    seq = book.army.seq
    texts = {
        "army": "aimed the attack at their soldiers",
        "workers": "sent the attack after their gatherers",
        "buildings": "aimed the attack at their buildings",
    }
    focus = book.army.focus
    text = (
        f"Our {book.army.action} on player {q} is out ({_army(me)} soldiers); now aimed at "
        f"{'whatever is nearest' if focus == 'any' else 'their ' + focus}. What should it hit first?"
    )

    def apply(value: str) -> list[Act]:
        if value not in options or book.army.seq != seq or book.army.focus == value:
            return []
        return [Focus(value)]

    return Open(choice("target", text, options), apply, texts)


def counter_question(me: PlayerSnapshot, book: OrderBook) -> Open | None:
    """A fight is on and a production queue is free: which of the soldier roles the strategist ordered to top up, by
    what the enemy fields (spec §4.6: the choices are limited to what the strategist's orders allow)."""
    if not me.idle_production or not (me.enemies_near_base > 0 or contact(me) >= CONTACT_MIN):
        return None
    describe = {
        "infantry_melee": "spearmen and swordsmen: hold the line, beat cavalry",
        "infantry_ranged": "archers and javelins: beat melee infantry from range",
        "cavalry": "riders: fast, beat ranged infantry and catch gatherers",
    }
    ordered = [r for r in COUNTER_ROLES if any(t.role == r for t in book.train)]
    options = {r: describe[r] for r in ordered if book.pending_count(r) < COUNTER_TOP}
    if not options:
        return None
    options["none"] = "no extra soldiers now"
    texts = {r: f"queued more {r} to counter what it saw" for r in ordered}
    seen = _enemy_soldiers(me)
    text = f"A fight is on; {len(me.idle_production)} production buildings are free. Enemy soldiers in sight: " + (
        ", ".join(f"{n} {r}" for r, n in seen.items()) or "none"
    )

    def apply(value: str) -> list[Act]:
        room = COUNTER_TOP - book.pending_count(value)
        if value not in ordered or room <= 0:
            return []
        return [Train(role=value, count=min(COUNTER_BATCH, room))]  # type: ignore[arg-type]

    return Open(choice("counter", text + ". Which ordered soldiers to add?", options), apply, texts)


def farmed(me: PlayerSnapshot) -> bool:
    return any(b.kind == "Field" and not b.foundation for b in me.buildings)


def exhausted_question(me: PlayerSnapshot, book: OrderBook, ran_out: list[str] | None = None) -> Open | None:
    """`ran_out`: the resources that went from some supply near the base to none (the reflex tracks them; a
    resource never seen near the base didn't run out). Food doesn't run out while the seat has fields."""
    split = dict(book.split)
    gone = [r for r in ran_out or () if split.get(r, 0) > 0 and not (r == "food" and farmed(me))]
    if not gone:
        return None
    out = gone[0]
    targets = [
        r
        for r in RESOURCES
        if r != out and split.get(r, 0) > 0 and (r not in me.nearby_resources or me.nearby_resources[r].amount > 0)
    ]
    if not targets:
        return None
    options = {r: f"move the {out} gatherers to {r}" for r in targets}
    options["keep"] = f"keep the split; {out} gatherers search farther away"

    def apply(value: str) -> list[Act]:
        if value == "keep" or value not in targets or book.split != split:  # the strategist changed it meanwhile
            return []
        new = dict(split)
        new[value] += new[out]
        new[out] = 0
        return [SetEconomy(**new)]

    text = f"{out} has run out near the base ({me.gatherers.get(out, 0)} gatherers on it)."
    texts = {r: f"moved the {out} gatherers to {r}" for r in targets}
    return Open(choice("exhausted", text, options), apply, texts, key=out)


def open_questions(
    me: PlayerSnapshot,
    book: OrderBook,
    *,
    standing: tuple[str, str] | None = None,
    ran_out: list[str] | None = None,
    quiet: set[str] = frozenset(),  # type: ignore[assignment]
) -> list[Open]:
    """`standing`: the strategist's own army order (default: the book's); `quiet`: questions resting after a change."""
    standing = standing or (book.army.action, book.army.target)
    made = {
        "threat": lambda: threat_question(me, book, standing),
        "regroup": lambda: regroup_question(me, book, standing),
        "engage": lambda: engage_question(me, book),
        "target": lambda: target_question(me, book),
        "counter": lambda: counter_question(me, book),
        "exhausted": lambda: exhausted_question(me, book, ran_out),
    }
    out = [make() for name, make in made.items() if name not in quiet]
    return [q for q in out if q is not None]


def open_decisions(me: PlayerSnapshot | None, book: OrderBook | None, events: list[Event]) -> list[str]:
    """What the seat can't act on until a brain answers (spec §5 `waiting_share`): an urgent event or a resource
    that ran out since the brain last read its brief, an idle army or enemies at the base, soldiers in a fight, or a
    free production queue with nothing ordered. A brain only counts as waited on while one of these is open."""
    out = []
    if any(e.urgent for e in events):
        out.append("urgent")
    if any(e.kind == "resource_exhausted" for e in events):
        out.append("exhausted")
    if me is not None and book is not None and me.state == "active":
        idle = sum(n for role, n in me.idle_units.items() if role not in ("villager", "other"))
        if idle >= IDLE_ARMY_MIN or me.enemies_near_base > 0:
            out.append("army")
        if contact(me) >= CONTACT_MIN:
            out.append("contact")
        if me.idle_production and not any(t.remaining > 0 and t.failure is None for t in book.train):
            out.append("production")
    return out


@dataclass
class ReflexStats:
    requests: int = 0
    answered: int = 0
    failed: int = 0
    skipped_busy: int = 0
    skipped_no_key: int = 0
    orders: int = 0
    questions: int = 0
    asked: Counter[str] = field(default_factory=Counter)
    by_kind: Counter[str] = field(default_factory=Counter)
    latencies_s: list[float] = field(default_factory=list)
    last_error: str | None = None

    def acted(self, kind: str) -> None:
        self.orders += 1
        self.by_kind[kind] += 1

    def summary(self, match_s: float) -> dict[str, Any]:
        minutes = match_s / 60
        p50, p95 = percentile(self.latencies_s, 50), percentile(self.latencies_s, 95)
        return {
            "requests": self.requests,
            "answered": self.answered,
            "failed": self.failed,
            "skipped_busy": self.skipped_busy,
            "skipped_no_key": self.skipped_no_key,
            "questions": self.questions,
            "asked": dict(self.asked),
            "orders": self.orders,
            "by_kind": dict(self.by_kind),
            "decisions_per_min": round(self.answered / minutes, 2) if minutes > 0 else 0.0,
            "latency_s": {
                "p50": None if p50 is None else round(p50, 3),
                "p95": None if p95 is None else round(p95, 3),
                "max": round(max(self.latencies_s), 3) if self.latencies_s else None,
            },
            "last_error": self.last_error,
        }


@dataclass(frozen=True)
class ReflexAction:
    """One answer that changed something: what the seat's action record keeps."""

    time_ms: int
    kind: str  # question:answer
    line: str  # for viewers
    what: str  # the orders (and focus) as tool calls
    status: str  # accepted, unless an order of it was not


OnAction = Callable[[ReflexAction], None]


class Reflex:
    """One seat's Luna reflex. `tick()` is called every interval by `run()`; it returns at once."""

    def __init__(
        self,
        player: int,
        book: OrderBook,
        submit: Callable[[_Order], Any],
        snapshot: Callable[[], Snapshot | None],
        metrics: SeatMetrics,
        decider: Decider | None = None,
        *,
        interval_s: float = INTERVAL_S,
        now_game_s: Callable[[], float] | None = None,
        on_action: OnAction | None = None,
    ):
        self.player = player
        self.on_action = on_action
        self.book = book
        self.submit = submit
        self.snapshot = snapshot
        self.metrics = metrics
        self.decider = decider
        self.interval_s = interval_s
        self.stats = ReflexStats()
        self._now_game_s = now_game_s or (lambda: (s.time_ms / 1000) if (s := snapshot()) is not None else 0.0)
        self._inflight: asyncio.Task[None] | None = None
        self._strategist_army: tuple[str, str] | None = None
        self._departure: int | None = None  # the book's army `seq` of the order the reflex itself put there
        self._had_supply: set[str] = set()
        self._ccs: set[int] | None = None  # the civic centres `nearby_resources` was summed around
        self._settled: set[str] = set()  # resources that ran out and whose question Luna already answered
        self._quiet_until: dict[str, int] = {}  # question -> game ms until which it rests after a change

    def _standing_army(self) -> tuple[str, str]:
        """The strategist's army order: the book's, unless the book still holds the reflex's own departure."""
        a = self.book.army
        if self._departure != a.seq:
            self._departure = None
            self._strategist_army = (a.action, a.target)
        assert self._strategist_army is not None
        return self._strategist_army

    def _ran_out(self, me: PlayerSnapshot) -> list[str]:
        """Resources that went from some supply near the base to none. When the base's civic centres change, the
        area the supply is summed over changed, so the totals start over (the same rule as the env's events)."""
        ccs = _base_ccs(me)
        if ccs != self._ccs:
            self._ccs = ccs
            self._had_supply = {r for r, near in me.nearby_resources.items() if near.amount > 0}
            self._settled.clear()
            return []
        out = []
        for r in RESOURCES:
            near = me.nearby_resources.get(r)
            if near is None:
                continue
            if near.amount > 0:
                self._had_supply.add(r)
                self._settled.discard(r)
            elif r in self._had_supply and r not in self._settled:
                out.append(r)
        return out

    def _now_ms(self) -> int:
        return int(self._now_game_s() * 1000)

    @property
    def busy(self) -> bool:
        return self._inflight is not None and not self._inflight.done()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            self.tick()

    def tick(self) -> None:
        """Open this interval's questions and send them, unless one request is still in flight."""
        if self.decider is None:
            self.stats.skipped_no_key += 1
            return
        if self.busy:
            self.stats.skipped_busy += 1
            return
        snap = self.snapshot()
        me = snap.players.get(self.player) if snap is not None else None
        if snap is None or me is None or me.state != "active":
            return
        now = snap.time_ms
        quiet = {name for name, until in self._quiet_until.items() if now < until}
        questions = open_questions(
            me, self.book, standing=self._standing_army(), ran_out=self._ran_out(me), quiet=quiet
        )
        if not questions:
            return
        context = SYSTEM + "\nEmpire now: " + json.dumps(json_brief(snap, self.player), separators=(",", ":"))
        context += "\nStanding orders: " + json.dumps(self.book.summary(), separators=(",", ":"))
        waiting = any(q.question.name in WAITED_ON for q in questions)
        self._inflight = asyncio.get_running_loop().create_task(self._ask(context, questions, waiting))

    async def _ask(self, context: str, questions: list[Open], waiting: bool) -> None:
        """`waiting`: the request answers a decision the seat can't act on until then (spec §5)."""
        assert self.decider is not None
        self.stats.requests += 1
        self.stats.questions += len(questions)
        self.stats.asked.update(q.question.name for q in questions)
        self.metrics.waits_reported = True  # the reflex reports its waits, so none is a measured zero
        if waiting:
            self.metrics.begin_waiting(self._now_game_s())
        started = time.perf_counter()
        try:
            decision = await self.decider.decide(context, [q.question for q in questions])
        except SpendCapReached as e:
            self.stats.last_error = str(e)
            self.decider = None  # stopped for the rest of the match
            return
        except Exception as e:  # noqa: BLE001 - a failed or slow reflex is skipped; the executor keeps going
            self.stats.failed += 1
            self.stats.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            log.info("reflex %s: request failed: %s", self.player, self.stats.last_error)
            return
        finally:
            if waiting:
                self.metrics.end_waiting(self._now_game_s())
        self.stats.answered += 1
        self.stats.latencies_s.append(time.perf_counter() - started)
        self.apply(questions, decision.answers)

    def apply(self, questions: list[Open], answers: dict[str, Answer]) -> None:
        now = self._now_ms()
        for q in questions:
            name = q.question.name
            answer = answers.get(name)
            if answer is None or answer.choice is None:
                continue
            if name == "exhausted" and q.key is not None:
                self._settled.add(q.key)
            changed = False
            whats, statuses = [], []
            for act in q.apply(answer.choice) or []:
                if isinstance(act, Focus):
                    self.book.army.focus = act.focus
                    changed = True
                    whats.append(f"focus({act.focus!r})")
                    continue
                result = self.submit(act)
                ok = bool(getattr(result, "ok", True))
                changed = changed or ok
                whats.append(call_text(act))
                statuses.append(str(getattr(result, "status", "accepted")))
                if isinstance(act, Army):
                    a = self.book.army
                    back = (a.action, a.target) == self._strategist_army
                    self._departure = None if back else a.seq
            if changed:
                kind = f"{name}:{answer.choice}"
                text = q.texts.get(answer.choice, f"{name}: {answer.choice}")
                self.stats.acted(kind)
                status = next((st for st in statuses if st != "accepted"), "accepted")
                if self.on_action is not None:
                    self.on_action(ReflexAction(now, kind, text, "; ".join(whats), status))
                if name in QUIET_MS:
                    self._quiet_until[name] = now + QUIET_MS[name]
                log.info("reflex %s: %s", self.player, text)

    async def stop(self) -> None:
        if self._inflight is not None and not self._inflight.done():
            self._inflight.cancel()
            await asyncio.gather(self._inflight, return_exceptions=True)
