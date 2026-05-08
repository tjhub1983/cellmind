# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
SC Engine: five-channel emotional regulation system.

Channels: fear, dopamine, oxytocin, endorphin, serotonin.

Usage:
    from cellmind import SCEngine

    sc = SCEngine()
    sc.receive_signal({"pheromone": "threat", "intensity": 0.8, "metadata": {"severity": 0.7}})
    fear = sc.get_fear_level()
"""

from typing import Dict, Any, List, Optional


class SCEngine:
    """
    Five-channel emotional regulation engine.

    Each channel has a `value` (0.0-1.0) driven by pheromone signals:
    - fear: threat signals
    - dopamine: reward signals
    - oxytocin: social bonding signals
    - endorphin: relief signals
    - serotonin: stability signals

    Usage:
        sc = SCEngine()
        sc.receive_signal({"pheromone": "threat", "intensity": 0.8})
        print(sc.get_fear_level())
    """

    def __init__(self) -> None:
        self._channels: Dict[str, float] = {
            "fear": 0.0,
            "dopamine": 0.0,
            "oxytocin": 0.0,
            "endorphin": 0.0,
            "serotonin": 0.0,
        }
        self._tick_count: int = 0

    def receive_signal(self, signal: Dict[str, Any]) -> None:
        """
        Receive a pheromone signal and update the corresponding channel.

        Args:
            signal: dict with keys:
                - pheromone (str): one of fear/dopamine/oxytocin/endorphin/serotonin
                  (also accepts: "threat" -> fear, "reward" -> dopamine)
                - intensity (float): signal strength (0.0-1.0)
                - metadata (dict, optional): additional context
        """
        pher = signal.get("pheromone", "fear")
        intensity = signal.get("intensity", 0.5)
        pher_map = {"threat": "fear", "reward": "dopamine", "bond": "oxytocin",
                    "relief": "endorphin", "stable": "serotonin"}
        channel = pher_map.get(pher, pher)
        if channel in self._channels:
            self._channels[channel] = min(1.0, self._channels[channel] + intensity * 0.3)
        self._tick_count += 1

    def tick(self) -> Dict[str, Any]:
        """Regulate channels on each tick (called every 30s in integrated mode)."""
        self._tick_count += 1
        for ch in self._channels:
            self._channels[ch] *= 0.95
            self._channels[ch] = max(0.0, self._channels[ch])
        return {
            "sc_tick": self._tick_count,
            "channels": {k: round(v, 4) for k, v in self._channels.items()},
        }

    def get_fear_level(self) -> float:
        """Get the current fear channel level (0.0-1.0)."""
        return self._channels.get("fear", 0.0)

    def get_dopamine_level(self) -> float:
        """Get the current dopamine channel level (0.0-1.0)."""
        return self._channels.get("dopamine", 0.0)

    def get_status(self) -> Dict[str, Any]:
        """Get the current engine status including all channel levels."""
        return {
            "tick_count": self._tick_count,
            "channels": {k: round(v, 4) for k, v in self._channels.items()},
        }


class REMWrapper:
    """
    REM sleep fragment storage layer.

    Stores memory fragments with semantic search and importance-based retention.
    Used for long-term episodic memory separate from cell memory.

    Usage:
        rem = REMWrapper()
        frag_id = rem.add_memory("discussed python memory system", importance=0.8)
        results = rem.search("python")
        stats = rem.get_stats()
    """

    def __init__(self) -> None:
        self._fragments: List[Dict[str, Any]] = []
        self._next_id: int = 1
        self._tick_count: int = 0

    def add_memory(
        self,
        content: str,
        source: str = "cellmind",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Add a memory fragment and return its ID."""
        tags = tags or []
        frag: Dict[str, Any] = {
            "id": f"frag_{self._next_id}",
            "content": content,
            "source": source,
            "importance": importance,
            "tags": tags,
            "created_at": self._tick_count,
        }
        self._fragments.append(frag)
        self._next_id += 1
        return frag["id"]

    def search(
        self,
        query: Optional[str] = None,
        tag: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search fragments by content keyword, tag, or minimum importance."""
        results: List[Dict[str, Any]] = []
        for f in self._fragments:
            if f["importance"] < min_importance:
                continue
            if tag and tag not in f.get("tags", []):
                continue
            if query:
                q = query.lower()
                if q in f["content"].lower() or any(q in t.lower() for t in f.get("tags", [])):
                    results.append(f)
            else:
                results.append(f)
        return sorted(results, key=lambda x: x["importance"], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get fragment statistics."""
        total = len(self._fragments)
        avg_imp = sum(f["importance"] for f in self._fragments) / total if total > 0 else 0
        return {
            "total_fragments": total,
            "avg_importance": round(avg_imp, 3),
        }

    def get_vitality(self) -> float:
        """System vitality = average importance."""
        if not self._fragments:
            return 1.0
        return sum(f["importance"] for f in self._fragments) / len(self._fragments)

    def tick(self) -> Dict[str, Any]:
        """Called periodically to update statistics."""
        self._tick_count += 1
        return {"rem_tick": self._tick_count, "stats": self.get_stats()}
