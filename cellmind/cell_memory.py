# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
CellMemory: biological cell-based memory with Hebbian learning.

Core concepts:
- Each concept becomes a Cell with strength (0.0-2.0)
- Repeated activation increases strength (Hebbian learning)
- Adjacent concepts form Hebbian connections
- Cells decay over time (natural forgetting)
- GlobalPool shares energy across all cells
- energy_box provides short-term buffer against decay

Usage:
    from cellmind import CellMemory

    cm = CellMemory()
    cell = cm.activate("python")
    print(cell.strength)  # 1.0
    cm.activate("python")
    print(cell.strength)  # 1.56
    cm.decay_all(days=7)
    print(cell.strength)  # decayed
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


# ============================================================
# Constants
# ============================================================

DECAY_RATE = 0.95
INITIAL_STRENGTH = 1.0
HEBB_STRENGTH = 0.2
CONNECTION_DECAY = 0.95
ELIMINATION_THRESHOLD = 0.01
MATCH_WEIGHT = 0.5
BOX_DECAY_RATE = 0.99
BOX_SAVE_RATIO = 0.20
BOX_MAX = 5.0
GLOBAL_POOL_CONTRIB = 0.10
GLOBAL_POOL_DECAY = 0.995
GLOBAL_POOL_MAX = 5.0
GLOBAL_POOL_INITIAL = 0.0
REPAY_MULTIPLIER = 1.5
MAX_CELLS = 100


# ============================================================
# Cell
# ============================================================

@dataclass
class Cell:
    cell_id: str
    preference: str
    strength: float
    energy_box: float
    global_debt: float = 0.0
    connections: Dict[str, float] = field(default_factory=dict)
    response_count: int = 0
    last_active: str = ""
    born: str = ""
    goal_tags: List[str] = field(default_factory=list)

    @staticmethod
    def create(preference: str) -> "Cell":
        now = datetime.now().isoformat()
        return Cell(
            cell_id=f"cell_{random.randint(10000, 99999)}",
            preference=preference,
            strength=INITIAL_STRENGTH,
            energy_box=0.0,
            global_debt=0.0,
            connections={},
            response_count=0,
            last_active=now,
            born=now,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Cell":
        return Cell(
            cell_id=d["cell_id"],
            preference=d["preference"],
            strength=d["strength"],
            energy_box=d.get("energy_box", 0.0),
            global_debt=d.get("global_debt", 0.0),
            connections=d.get("connections", {}),
            response_count=d.get("response_count", 0),
            last_active=d.get("last_active", ""),
            born=d.get("born", ""),
            goal_tags=d.get("goal_tags", []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# GlobalPool
# ============================================================

class GlobalPool:
    """Shared energy pool across all cells."""

    def __init__(self, initial: float = GLOBAL_POOL_INITIAL):
        self.energy: float = initial
        self.max_energy: float = GLOBAL_POOL_MAX

    def contribute(self, amount: float) -> float:
        room = self.max_energy - self.energy
        stored = min(amount, room)
        self.energy += stored
        return stored

    def draw(self, amount: float) -> float:
        available = min(self.energy, amount)
        self.energy -= available
        return available

    def decay_all(self, days: float = 1.0):
        self.energy *= GLOBAL_POOL_DECAY ** days
        self.energy = max(0.0, self.energy)

    def to_dict(self) -> dict:
        return {"energy": self.energy, "max_energy": self.max_energy}

    @staticmethod
    def from_dict(d: dict) -> "GlobalPool":
        return GlobalPool(initial=d.get("energy", GLOBAL_POOL_INITIAL))

    def __repr__(self):
        return f"GlobalPool(energy={self.energy:.3f}/{self.max_energy})"


# ============================================================
# CellMemory
# ============================================================

class CellMemory:
    """
    Cell-based memory with Hebbian learning.

    Cells store conceptual preferences with associated strength values.
    Repeated activation increases strength via Hebbian learning.
    Cells form connections when activated in sequence.
    All cells share a GlobalPool for energy redistribution.

    Usage:
        cm = CellMemory()
        cm.activate("python")
        cm.activate("python")
        top = cm.get_top_cells(5)
        cm.decay_all(days=7)
        cm.save("state.json")
        cm.load("state.json")

    Attributes:
        cells: Dict[str, Cell] — all active cells
        global_pool: GlobalPool — shared energy pool
    """

    def __init__(self, save_path: Optional[str] = None):
        self.cells: Dict[str, Cell] = {}
        self.global_pool = GlobalPool(initial=GLOBAL_POOL_INITIAL)
        self.save_path = save_path
        if save_path:
            self._load()

    # ---- Persistence ----

    def save(self, path: Optional[str] = None):
        path = path or self.save_path
        if not path:
            raise ValueError("save_path not set")
        data = {k: v.to_dict() for k, v in self.cells.items()}
        data["_global_pool"] = self.global_pool.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if self.save_path and __import__("os").path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pool_data = data.pop("_global_pool", {})
                self.cells = {}
                for k, v in data.items():
                    v.setdefault("energy_box", 0.0)
                    v.setdefault("global_debt", 0.0)
                    self.cells[k] = Cell.from_dict(v)
                self.global_pool = GlobalPool.from_dict(pool_data)
            except Exception:
                pass

    # ---- Core operations ----

    def activate(self, preference: str) -> Cell:
        """
        Activate a cell by preference string.

        If cell exists: apply Hebbian boost (70% to strength, 20% to energy_box,
        10% to GlobalPool), repay debt if any.
        If cell is new: create it with INITIAL_STRENGTH.

        Returns:
            Cell — the activated cell
        """
        candidates = [c for c in self.cells.values() if c.preference == preference]
        if not candidates:
            cell = self._get_or_create(preference)
            self.global_pool.contribute(HEBB_STRENGTH * GLOBAL_POOL_CONTRIB)
            cell.response_count = 1
            cell.last_active = datetime.now().isoformat()
            return cell

        def score(c: Cell) -> float:
            match = len(set(c.preference) & set(preference)) / max(len(c.preference), 1)
            return MATCH_WEIGHT * match + (1 - MATCH_WEIGHT) * (c.strength / INITIAL_STRENGTH)

        candidates.sort(key=score, reverse=True)
        winner = candidates[0]

        boost = HEBB_STRENGTH
        to_box = boost * BOX_SAVE_RATIO
        to_pool = boost * GLOBAL_POOL_CONTRIB
        to_strength_raw = boost - to_box - to_pool

        net_boost = to_strength_raw
        if winner.global_debt > 0:
            repay = min(winner.global_debt, net_boost)
            winner.global_debt -= repay
            net_boost -= repay

        self.global_pool.contribute(to_pool)
        winner.strength = min(2.0, winner.strength + net_boost)
        winner.energy_box = min(BOX_MAX, winner.energy_box + to_box)
        winner.response_count += 1
        winner.last_active = datetime.now().isoformat()
        self._enforce_capacity()
        return winner

    def _get_or_create(self, preference: str) -> Cell:
        for cell in self.cells.values():
            if cell.preference == preference:
                return cell
        cell = Cell.create(preference)
        self.cells[cell.cell_id] = cell
        self._enforce_capacity()
        return cell

    def _enforce_capacity(self):
        if len(self.cells) > MAX_CELLS:
            self.eliminate_weak()

    def hebb_connect(self, cell_a: Cell, cell_b: Cell):
        """Form bidirectional Hebbian connection between two cells."""
        if cell_a.cell_id == cell_b.cell_id:
            return
        current_ab = cell_a.connections.get(cell_b.cell_id, 0.0)
        cell_a.connections[cell_b.cell_id] = min(1.0, current_ab + HEBB_STRENGTH)
        current_ba = cell_b.connections.get(cell_a.cell_id, 0.0)
        cell_b.connections[cell_a.cell_id] = min(1.0, current_ba + HEBB_STRENGTH)

    def activate_tokens(self, tokens: List[str]):
        """
        Activate a sequence of tokens and form Hebbian connections between
        cells that existed before this activation (old+old only).

        Args:
            tokens: List of preference strings
        """
        pre_existing = set(self.cells.keys())
        prev = None
        for token in tokens:
            cell = self.activate(token)
            if prev and prev.cell_id in pre_existing and cell.cell_id in pre_existing:
                self.hebb_connect(prev, cell)
            prev = cell

    # ---- Decay ----

    def decay_all(self, days: float = 1.0):
        """
        Apply time-based decay to all cells and the GlobalPool.

        Compensation chain (3 levels):
        1. energy_box absorbs decay
        2. GlobalPool absorbs remainder (records as debt)
        3. Remaining loss deducted from strength

        Args:
            days: Number of simulated days to decay
        """
        box_decay = BOX_DECAY_RATE ** days
        strength_decay = DECAY_RATE ** days

        for cell in list(self.cells.values()):
            strength_loss = cell.strength * (1 - strength_decay)

            if cell.energy_box > 0 and strength_loss > 0:
                compensated = min(cell.energy_box, strength_loss)
                cell.energy_box -= compensated
                strength_loss -= compensated

            if strength_loss > 0 and self.global_pool.energy > 0:
                compensated = self.global_pool.draw(strength_loss)
                if compensated > 0:
                    cell.global_debt += compensated * REPAY_MULTIPLIER
                    strength_loss -= compensated

            cell.strength -= strength_loss
            cell.strength = max(0.0, cell.strength)
            cell.energy_box *= box_decay

        self.global_pool.decay_all(days=days)

        for cell in list(self.cells.values()):
            if cell.strength < ELIMINATION_THRESHOLD:
                deficit = ELIMINATION_THRESHOLD - cell.strength
                drawn = self.global_pool.draw(deficit)
                if drawn > 0:
                    cell.strength += drawn
                if cell.strength < ELIMINATION_THRESHOLD:
                    self.eliminate_weak(cell_id=cell.cell_id)

    def eliminate_weak(self, cell_id: Optional[str] = None) -> int:
        """
        Eliminate cells below strength threshold, or specific cell if cell_id given.
        Also enforces MAX_CELLS by removing weakest cells.

        Returns:
            int — number of cells eliminated
        """
        before = len(self.cells)

        if cell_id and cell_id in self.cells:
            if self.cells[cell_id].strength < ELIMINATION_THRESHOLD:
                del self.cells[cell_id]
            return before - len(self.cells)

        to_remove = [cid for cid, c in self.cells.items()
                     if c.strength < ELIMINATION_THRESHOLD]

        if len(self.cells) > MAX_CELLS:
            sorted_cells = sorted(self.cells.items(), key=lambda x: x[1].strength)
            excess = len(self.cells) - MAX_CELLS
            for cid, _ in sorted_cells[:excess]:
                if cid not in to_remove:
                    to_remove.append(cid)

        for cid in to_remove:
            del self.cells[cid]

        return before - len(self.cells)

    # ---- Queries ----

    def get_top_cells(self, n: int = 10) -> List[Cell]:
        """Return top N cells sorted by strength."""
        return sorted(self.cells.values(), key=lambda c: c.strength, reverse=True)[:n]

    def build_context_prompt(self, top_n: int = 8) -> str:
        """
        Build a text summary of current cell state for system prompt injection.

        Args:
            top_n: Number of top cells to include

        Returns:
            str — formatted cell state summary
        """
        top = self.get_top_cells(top_n)
        if not top:
            return ""
        pool_pct = self.global_pool.energy / self.global_pool.max_energy * 100
        lines = [
            "[Internal Cell State]",
            f"# Active cells: {len(top)} / {len(self.cells)} total | "
            f"GlobalPool: {self.global_pool.energy:.3f}/{self.global_pool.max_energy} ({pool_pct:.0f}%)",
            "# Format: preference | strength | box | debt | responses",
        ]
        for c in top:
            conn_prefs = []
            for cid, w in c.connections.items():
                if w > 0:
                    other = self.cells.get(cid)
                    if other:
                        conn_prefs.append(f"{other.preference}({w:.2f})")
            conn_str = ", ".join(conn_prefs[:3]) if conn_prefs else "none"
            debt_str = f" debt={c.global_debt:.2f}" if c.global_debt > 0 else ""
            lines.append(
                f"- {c.preference} | s={c.strength:.2f} | "
                f"box={c.energy_box:.3f}{debt_str} | conn={conn_str} | resp={c.response_count}"
            )
        lines.append("[End Internal State]")
        return "\n".join(lines)

    def pool_status(self) -> str:
        return f"pool={self.global_pool.energy:.3f}/{self.global_pool.max_energy}"

    def to_dict(self) -> dict:
        """Serialize CellMemory to dict for persistence."""
        data = {k: v.to_dict() for k, v in self.cells.items()}
        data["_global_pool"] = self.global_pool.to_dict()
        return data

    @staticmethod
    def from_dict(data: dict) -> "CellMemory":
        """Deserialize CellMemory from dict."""
        cm = CellMemory()
        pool_data = data.pop("_global_pool", {})
        cm.cells = {}
        for k, v in data.items():
            v.setdefault("energy_box", 0.0)
            v.setdefault("global_debt", 0.0)
            v.setdefault("goal_tags", [])
            cm.cells[k] = Cell.from_dict(v)
        cm.global_pool = GlobalPool.from_dict(pool_data)
        return cm

    def __repr__(self):
        return f"CellMemory(cells={len(self.cells)}, {self.pool_status()})"