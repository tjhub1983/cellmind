# CellMind

> AI memory architecture inspired by biological cell memory.

CellMind solves the fundamental problem every AI system faces: **context loss across sessions**. While LLMs forget after each conversation, CellMind uses a biological cell-based memory model that persists, learns, and grows.

---

## What It Does

CellMind treats memory as **cells** — each concept becomes a cell with a strength value. Repeated activation increases strength via Hebbian learning. Cells form connections when concepts appear together. Natural decay handles forgetting. A shared GlobalPool redistributes energy across all cells.

This is not a vector database. This is a living memory system that mirrors biological neurons.

---

## Quick Start

```bash
pip install cellmind
```

```python
from cellmind import CellMindCore

cm = CellMindCore()

# Discuss anything — CellMind learns from your conversations
cm.discuss_text("I am working on a python project")

# Check what's in memory
status = cm.get_status()
print(f"Cells: {status['cells_total']}")
print(f"Top concepts: {[c['preference'] for c in status['top_cells']]}")

# Set goals — tracked and pursued over time
goal = cm.set_goal("write better python code", priority=0.8)
cm.pursue_goal(goal.goal_id)

# Get context for system prompt injection
context = cm.get_context_prompt()
print(context)
```

---

## Core Concepts

### Cell Memory

Each concept you discuss becomes a `Cell` with:
- `strength` (0.0–2.0): how well the concept is remembered
- `energy_box`: short-term buffer against decay
- `connections`: Hebbian links to related cells
- `goal_tags`: goals this cell contributes to

### Hebbian Learning

When you discuss "python" 3 times, its cell's strength grows: 1.0 → 1.56 → 2.0 (capped). When "python" and "programming" appear together, a Hebbian connection forms.

### Decay + Compensation

Cells decay over time (natural forgetting). The system has 3 compensation levels:
1. `energy_box` absorbs decay first
2. `GlobalPool` provides secondary rescue (records debt)
3. Remaining loss deducted from `strength`

### Five-Channel Emotional System

SC Engine models five chemical channels:
- **Fear**: threat detection and self-preservation
- **Dopamine**: reward signals and motivation
- **Oxytocin**: social bonding
- **Endorphin**: pain relief
- **Serotonin**: stability and mood regulation

### REM Sleep Layer

Long-term episodic memory stored as fragments. Searchable by content, tags, and importance.

### Goal System

Goals are decomposed into steps. Each step activates relevant cells. Progress tracked and persisted.

---

## Architecture

```
User Input
    ↓
Token Extraction (stopword-filtered, min 3 chars)
    ↓
Cell Activation (Hebbian boost: 70% strength / 20% energy_box / 10% GlobalPool)
    ↓
Working Memory (5-item focus, fast decay)
    ↓
Context Prompt Generation (for LLM system injection)
    ↓
State Persistence (~/.cellmind/cellmind_state.json)
```

Additional layers:
- **SC Engine**: five-channel emotional regulation (tick every 30s)
- **REM**: fragment storage with semantic search
- **Decay**: triggered on time passage, applied to all cells and GlobalPool

---

## Installation

```bash
pip install cellmind
```

Requires Python 3.8+

---

## API Reference

### CellMindCore

```python
from cellmind import CellMindCore

cm = CellMindCore()  # loads existing state or creates fresh
```

**Core methods:**

| Method | Description |
|--------|-------------|
| `discuss_text(text)` | Process discussion: tokens → cells → emotion → history |
| `activate_text(text)` | Activate tokens without sentiment detection |
| `set_goal(description, priority)` | Create goal with auto-generated steps |
| `pursue_goal(goal_id)` | Execute pending goal steps |
| `get_status()` | Full system status dict |
| `get_context_prompt()` | System prompt injection string |
| `save()` | Persist state to disk |

**State:**

```python
cm.cell_memory    # CellMemory — long-term cell storage
cm.working_memory # WorkingMemory — 5-item focus
cm.emotion        # EmotionState — current valence/arousal
cm.goals          # List[Goal] — active goals
```

### CellMemory (standalone)

```python
from cellmind import CellMemory

cm = CellMemory()
cm.activate("python")           # create or boost cell
cm.activate("python")           # strength: 1.0 → 1.56
cm.activate_tokens(["python", "programming"])  # sequential + Hebbian connect
cm.decay_all(days=7)            # apply 7 days of decay
cm.get_top_cells(10)            # sorted by strength
cm.save("state.json")           # persist
```

### SCEngine

```python
from cellmind import SCEngine

sc = SCEngine()
sc.receive_signal({"pheromone": "threat", "intensity": 0.8})
print(sc.get_fear_level())  # 0.24
sc.tick()                   # regulation tick
print(sc.get_status())
```

### REMWrapper

```python
from cellmind import REMWrapper

rem = REMWrapper()
rem.add_memory("discussed python design", importance=0.8, tags=["python"])
results = rem.search("python")
print(rem.get_stats())
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HEBB_STRENGTH` | 0.2 | Boost per activation |
| `INITIAL_STRENGTH` | 1.0 | New cell starting strength |
| `MAX_CELLS` | 100 | Cell count cap |
| `DECAY_RATE` | 0.95 | Daily strength retention |
| `BOX_SAVE_RATIO` | 0.20 | Boost fraction to energy_box |
| `GLOBAL_POOL_CONTRIB` | 0.10 | Boost fraction to GlobalPool |
| `WM_CAPACITY` | 5 | Working memory items |

---

## License

Apache 2.0 — commercial use allowed, no patents restricted.

---

*"Testing is not proving the code works. Testing is proving the physics is correct."*
— CellMind Musk Team