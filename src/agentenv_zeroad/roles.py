"""Map the order vocabulary's roles and structure names to a civ's real templates, read from the engine (spec §4.5).

Nothing here is a hand-typed unit list. For a civ the catalog asks the engine (`/templates`, which returns each
template fully merged with its parents) for:
1. the civ's worker (`units/<civ>/support_female_citizen`, else what the civic centre trains as a villager) and its
   `Builder/Entities`: every structure the civ can build;
2. each structure's `Trainer/Entities`: every unit it can train;
3. each unit's `Identity` classes, which give its role with the same rules as the snapshot query (`js/snapshot.js`).

Template lists use `{civ}` and `{native}` placeholders and list entries that only some civs have. Only the names that
exist are asked for (`existing_templates`, which loads nothing): the engine logs an error for each unknown name it is
asked to load, and a rendered match shows those on screen. What remains is exactly the civ's set. A structure's
order name is its template basename (`structures/rome/barracks` -> "barracks"). Tables are cached per civ for the
life of the catalog.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agentenv_zeroad.engine import GameEngine
from agentenv_zeroad.state import RESOURCES, ROLES

STRUCTURE_KINDS = (
    "CivCentre", "House", "Barracks", "Stable", "Range", "ArcheryRange", "Arsenal", "Workshop", "Fortress", "Temple",
    "Market", "Storehouse", "Farmstead", "Field", "Corral", "Dock", "Blacksmith", "Forge", "Tower", "Wall", "Gate",
    "Wonder", "Gymnasium", "ElephantStable", "Theater", "Palisade",
)  # fmt: skip
# Every civ selectable in 0 A.D. 0.28 (`simulation/data/civs/*.json` with `SelectableInGameSetup`).
CIVS = (
    "athen", "brit", "cart", "gaul", "germ", "han", "iber", "kush", "mace", "maur", "pers", "ptol", "rome", "sele",
    "spart",
)  # fmt: skip


def role_of(classes: Iterable[str]) -> str:
    """The order role for a unit's classes; mirrors `role()` in js/snapshot.js."""
    cls = set(classes)
    if "Siege" in cls:
        return "siege"
    if "Hero" in cls:
        return "other"
    if "Champion" in cls:
        return "champion"
    if cls & {"Ship", "Trader", "Healer", "Animal"}:
        return "other"
    if "Cavalry" in cls:
        return "cavalry"
    if "Infantry" in cls:
        return "infantry_ranged" if "Ranged" in cls else "infantry_melee"
    if cls & {"FemaleCitizen", "Worker"}:
        return "villager"
    return "other"


def kind_of(classes: Iterable[str]) -> str:
    cls = set(classes)
    return next((k for k in STRUCTURE_KINDS if k in cls), "Other")


@dataclass(frozen=True)
class Template:
    name: str
    classes: frozenset[str]
    cost: dict[str, int]
    requires: tuple[str, ...] = ()
    trains: tuple[str, ...] = ()
    builds: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    population: int = 0

    @property
    def key(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def role(self) -> str:
        return role_of(self.classes)

    @property
    def kind(self) -> str:
        return kind_of(self.classes)


def parse_template(name: str, xml: str, civ: str) -> Template:
    """Parse one merged template as `/templates` returns it (a fragment of elements without a root)."""
    root = ET.fromstring(f"<Entity>{xml}</Entity>")

    def tokens(path: str) -> list[str]:
        node = root.find(path)
        return _expand(node.text.split() if node is not None and node.text else [], civ)

    classes = frozenset(tokens("Identity/Classes") + tokens("Identity/VisibleClasses"))
    cost = {r: int(float(root.findtext(f"Cost/Resources/{r}") or 0)) for r in RESOURCES}
    return Template(
        name=name,
        classes=classes,
        cost={r: v for r, v in cost.items() if v},
        requires=tuple(tokens("Identity/Requirements/Techs")),
        trains=tuple(tokens("Trainer/Entities")),
        builds=tuple(tokens("Builder/Entities")),
        drop=tuple(tokens("ResourceDropsite/Types")),
        population=int(float(root.findtext("Cost/Population") or 0)),
    )


def _expand(names: Sequence[str], civ: str) -> list[str]:
    return [n.replace("{civ}", civ).replace("{native}", civ) for n in names if not n.startswith("-")]


@dataclass
class RoleTable:
    """One civ's templates: structures by order name and trainable units by role."""

    civ: str
    structures: dict[str, Template]
    units: dict[str, Template]
    by_role: dict[str, list[Template]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.by_role:
            for unit in self.units.values():
                self.by_role.setdefault(unit.role, []).append(unit)
            for units in self.by_role.values():
                units.sort(key=lambda u: (len(u.requires), sum(u.cost.values()), u.name))

    @property
    def villager(self) -> Template | None:
        units = self.by_role.get("villager", [])
        return next((u for u in units if "FemaleCitizen" in u.classes), units[0] if units else None)

    def roles(self) -> list[str]:
        return [r for r in ROLES if self.by_role.get(r)]

    def templates_for(self, role: str) -> list[str]:
        return [u.name for u in self.by_role.get(role, [])]

    def structure(self, key: str) -> Template | None:
        return self.structures.get(key)

    def structure_of_kind(self, kind: str) -> Template | None:
        return next((s for s in self.structures.values() if s.kind == kind), None)

    def trained_at(self, role: str) -> list[str]:
        """Structure names that can train some unit of `role`."""
        names = set(self.templates_for(role))
        return sorted(s.key for s in self.structures.values() if names & set(s.trains))


class RoleCatalog:
    """Builds and caches `RoleTable`s from an engine's templates."""

    def __init__(self, engine: GameEngine):
        self.engine = engine
        self._tables: dict[str, RoleTable] = {}

    async def table(self, civ: str) -> RoleTable:
        if civ not in self._tables:
            self._tables[civ] = await self._build(civ)
        return self._tables[civ]

    async def _build(self, civ: str) -> RoleTable:
        cc_name = f"structures/{civ}/civil_centre"
        seed = await self._fetch([f"units/{civ}/support_female_citizen", cc_name], civ)
        worker = seed.get(f"units/{civ}/support_female_citizen")
        if worker is None and cc_name in seed:
            # Civs without a female citizen (germ trains `support_civilian`): the civic centre's villager builds.
            trained = await self._fetch(list(seed[cc_name].trains), civ)
            worker = next((u for u in trained.values() if u.role == "villager" and u.builds), None)
        builds = list(worker.builds) if worker else []
        if cc_name not in builds:
            builds.insert(0, cc_name)
        structures = await self._fetch([b for b in dict.fromkeys(builds) if b.startswith("structures/")], civ)
        trainable = list(dict.fromkeys(u for s in structures.values() for u in s.trains))
        units = await self._fetch(trainable, civ)
        if worker:
            units.setdefault(worker.name, worker)
        return RoleTable(
            civ=civ,
            structures={s.key: s for s in structures.values() if "Structure" in s.classes},
            units={u.name: u for u in units.values() if "Unit" in u.classes},
        )

    async def _fetch(self, names: Sequence[str], civ: str) -> dict[str, Template]:
        found = await self.engine.templates(await self.engine.existing_templates(list(names)))
        return {name: parse_template(name, xml, civ) for name, xml in found.items()}
