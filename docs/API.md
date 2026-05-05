# CellMind API Reference

## Installation

```bash
pip install cellmind
```

---

## `CellMindCore`

Main entry point. Initializes from saved state or creates fresh.

```python
from cellmind import CellMindCore

cm = CellMindCore(state_dir="~/.cellmind")
```

### Methods

#### `discuss_text(text: str) -> dict`

Process a discussion turn. Activates tokens, updates emotion, appends to history.

```python
result = cm.discuss_text("I love programming in python")
# Returns:
# {
#   "tokens_activated": ["love", "programming", "python"],
#   "sentiment": "positive",
#   "emotion": "excited(v=0.42, a=0.60)",
#   "wm_focus": ["python", "programming"],
#   "top_cells": ["python", "programming", "love"]
# }
```

#### `activate_text(text: str, goal_id: str = None, boost_override: float = None) -> List[str]`

Activate tokens without sentiment detection. Useful for programmatic activation.

```python
tokens = cm.activate_text("python memory system", goal_id="goal_1")
# Returns: ["python", "memory", "system"]
```

#### `set_goal(description: str, priority: float = 0.5) -> Goal`

Create a new goal. Steps auto-generated from description tokens.

```python
goal = cm.set_goal("learn python programming", priority=0.8)
# goal.goal_id = "goal_1"
# goal.steps = [GoalStep(...), GoalStep(...)]
```

#### `pursue_goal(goal_id: str) -> dict`

Execute all pending steps in a goal. Activates relevant cells, marks steps complete.

```python
result = cm.pursue_goal("goal_1")
# Returns:
# {
#   "goal_id": "goal_1",
#   "progress": 1.0,
#   "results": ["Explored: python", "Explored: programming"]
# }
```

#### `get_status() -> dict`

Full system status.

```python
status = cm.get_status()
# {
#   "cells_total": 15,
#   "top_cells": [{"preference": "python", "strength": 1.56, ...}],
#   "wm_items": [{"preference": "python", "activation": 1.4, ...}],
#   "global_pool": 0.23,
#   "emotion": "neutral(v=0.00, a=0.50)",
#   "active_goals": [...],
#   "conversation_count": 12
# }
```

#### `get_context_prompt() -> str`

System prompt injection string for LLM context.

```python
context = cm.get_context_prompt()
# "=== CellMind Memory State ===\nCells: 15 total | Top: python, programming..."
```

#### `save()`

Persist current state to `~/.cellmind/cellmind_state.json`.

---

## `CellMemory`

Long-term cell memory. Can be used standalone.

```python
from cellmind import CellMemory

cm = CellMemory(save_path="state.json")
```

### Methods

#### `activate(preference: str) -> Cell`

Activate or create cell by preference string.

```python
cell = cm.activate("python")
# cell.strength = 1.0 (first time) or boosted (subsequent)
```

#### `activate_tokens(tokens: List[str])`

Activate multiple tokens and form Hebbian connections.

```python
cm.activate_tokens(["python", "programming", "memory"])
```

#### `decay_all(days: float = 1.0)`

Apply time-based decay to all cells and GlobalPool.

```python
cm.decay_all(days=7)  # simulate 7 days
```

#### `get_top_cells(n: int = 10) -> List[Cell]`

Return top N cells sorted by strength.

#### `save(path: str = None)`

Persist to JSON file.

---

## `SCEngine`

Five-channel emotional regulation engine.

```python
from cellmind import SCEngine

sc = SCEngine()
```

### Methods

#### `receive_signal(signal: dict)`

Inject pheromone signal.

```python
sc.receive_signal({
    "pheromone": "threat",   # fear/dopamine/oxytocin/endorphin/serotonin
    "intensity": 0.8,        # 0.0-1.0
    "metadata": {"severity": 0.7}
})
```

#### `tick() -> dict`

Regulation tick. Call every 30s in integrated mode.

#### `get_fear_level() -> float`

Get current fear channel value.

#### `get_status() -> dict`

Get all channel values.

---

## `REMWrapper`

REM fragment storage layer.

```python
from cellmind import REMWrapper

rem = REMWrapper()
```

### Methods

#### `add_memory(content: str, source: str = "cellmind", importance: float = 0.5, tags: list = None) -> str`

Add fragment and return ID.

```python
frag_id = rem.add_memory("discussed python design", importance=0.8, tags=["python"])
```

#### `search(query: str = None, tag: str = None, min_importance: float = 0.0) -> list`

Search fragments.

```python
results = rem.search("python", min_importance=0.5)
```

#### `get_stats() -> dict`

Statistics.

```python
print(rem.get_stats())
# {"total_fragments": 12, "avg_importance": 0.65}
```

---

## `EmotionState`

Emotional state with VAD model.

```python
from cellmind import EmotionState

em = EmotionState()
em.apply_sentiment(0.5)  # positive
em.classify()            # "excited"
em.status()              # "excited(v=0.25, a=0.60)"
```

---

## Data Classes

### `Cell`

```python
@dataclass
class Cell:
    cell_id: str
    preference: str
    strength: float        # 0.0-2.0
    energy_box: float
    global_debt: float
    connections: Dict[str, float]   # cell_id -> weight
    response_count: int
    last_active: str
    born: str
    goal_tags: List[str]
```

### `Goal`

```python
@dataclass
class Goal:
    goal_id: str
    description: str
    priority: float        # 0.0-1.0
    state: str            # pending/active/completed
    progress: float       # 0.0-1.0
    steps: List[GoalStep]
    created_at: str
    completed_at: Optional[str]
```

### `GoalStep`

```python
@dataclass
class GoalStep:
    step_id: str
    description: str
    topic: str
    state: str
    attempts: int
    result: str
```

---

## State Persistence

State saved to `~/.cellmind/cellmind_state.json` as a single JSON file. Load is automatic on `CellMindCore()` init.

For manual control:

```python
cm.save()  # explicit save

# Load CellMemory directly
from cellmind import CellMemory
cm = CellMemory.from_dict(data)  # if you have raw dict
```

---

## Exceptions

CellMind does not throw custom exceptions. All errors propagate as standard Python exceptions (KeyError, ValueError, IOError). The system is designed to degrade gracefully — if state file is corrupted, it starts fresh and logs a warning.