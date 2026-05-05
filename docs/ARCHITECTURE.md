# CellMind Architecture

## Design Philosophy

CellMind mirrors biological neuron memory: cells store concepts, activation strengthens connections, decay handles forgetting, and a shared energy pool redistributes resources.

The goal: AI that remembers across sessions the way humans do — not as a database, but as a living system that learns from experience.

---

## Memory Architecture

### Cell Memory (LTM)

```
Concept → Cell(preference, strength, energy_box, connections)
```

**Activation flow:**
1. Exact match → boost existing cell
2. Partial match → score by match ratio + strength, boost winner
3. No match → create new cell at `strength=1.0`

**Boost distribution (per activation):**
```
boost = 0.2 (HEBB_STRENGTH)
  ├── 70% → strength (net, after debt repayment)
  ├── 20% → energy_box (decay buffer)
  └── 10% → GlobalPool (shared reserve)
```

**Debt repayment:**
When a cell has `global_debt > 0`, the net boost first repays debt before increasing strength. This creates a "欠债-还债" cycle that distributes energy efficiently.

**Hebbian connections:**
When two cells are activated in sequence (and both existed before the activation), a bidirectional connection forms. Connection weight = 0.2, capped at 1.0.

**Decay with 3-level compensation:**
```
strength_loss = strength * (1 - DECAY_RATE^days)

Level 1: energy_box absorbs
Level 2: GlobalPool absorbs + records debt
Level 3: remaining loss from strength

If strength < 0.01 after compensation → cell eliminated
```

**MAX_CELLS = 100:**
When capacity exceeded, weakest cells removed. Preference: eliminate cells below threshold first, then fill quota with weakest cells.

---

### GlobalPool

Shared energy reserve across all cells. Decays at 0.995/day (extremely slow). When a cell's energy_box is depleted and decay hits strength, GlobalPool provides rescue but records debt (REPAY_MULTIPLIER = 1.5×).

---

### Working Memory (WM)

5-item capacity (Miller's 7±2, conservative). Fast decay rate (0.85/cycle vs LTM 0.95/day).

**Retrieval flow:**
1. Search LTM for cells matching query
2. Score by (match_ratio, strength)
3. If already in WM: refresh activation
4. If not in WM and full: evict weakest
5. Add with WM_RETRIEVAL_BOOST = 0.4

---

### Emotion System

VAD model (Valence-Arousal-Dominance). Four emotional channels derived from VAD values. Emotion influences activation boost: positive emotions increase boost, negative decrease.

---

### Goal System

Goals have auto-generated steps from token extraction. Steps activate relevant cells when pursued. Progress tracked as step completion ratio. Goals persist across sessions.

---

### SC Five-Channel Engine

Biological pheromone model. Each channel driven by signal injection:
- Fear: threat signals
- Dopamine: reward signals
- Oxytocin: bonding signals
- Endorphin: relief signals
- Serotonin: stability signals

Each tick (30s): all channels decay by 5%. New signals increment channel values.

---

### REM Fragment Layer

Episodic memory separate from cell memory. Stored as fragments with content, importance (0.0–1.0), source, and tags. Searchable by keyword, tag filter, minimum importance. Used for long-term memory that survives cell decay cycles.

---

## Data Flow

```
User Input (text)
    ↓
Token Extraction (regex, stopword filter, min 3 chars)
    ↓
CellMemory.activate_tokens()
    ├── Cell.activate() × N tokens
    ├── Hebbian connections for pre-existing pairs
    └── WorkingMemory.retrieve()
    ↓
EmotionState (sentiment detection)
    ↓
Conversation history append
    ↓
CellMindCore.save() (auto or manual)
    ↓
~/.cellmind/cellmind_state.json
```

Context prompt generation: top cells by strength → formatted string → injected into LLM system prompt.

---

## State Persistence

Single JSON file at `~/.cellmind/cellmind_state.json`:
```json
{
  "cell_memory": { "cell_id": Cell.to_dict(), "_global_pool": {...} },
  "working_memory": { "capacity": 5, "items": {...} },
  "emotion": { "valence": 0.3, "arousal": 0.5, "dominance": 0.5 },
  "goals": [Goal.to_dict()],
  "completed_goals": [Goal.to_dict()],
  "conversation_history": [...]
}
```

Load on `CellMindCore()` init. Save on `save()` call or after each major operation.

---

## Constants Reference

```python
HEBB_STRENGTH = 0.2        # activation boost
INITIAL_STRENGTH = 1.0     # new cell strength
MAX_CELLS = 100            # cell count cap
DECAY_RATE = 0.95          # daily strength retention (5% lost)
BOX_DECAY_RATE = 0.99      # energy_box daily retention (1% lost)
BOX_SAVE_RATIO = 0.20      # boost → energy_box
GLOBAL_POOL_CONTRIB = 0.10  # boost → GlobalPool
REPAY_MULTIPLIER = 1.5     # debt recording multiplier
WM_CAPACITY = 5            # working memory items
WM_DECAY_RATE = 0.85       # working memory decay
WM_RETRIEVAL_BOOST = 0.4   # retrieval activation bonus
EMOTION_DECAY_RATE = 0.92  # valence decay rate
```