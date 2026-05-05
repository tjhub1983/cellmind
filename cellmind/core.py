# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
CellMindCore: full integration of CellMemory + WorkingMemory + Emotion + Goals.

Usage:
    from cellmind import CellMindCore

    cm = CellMindCore()
    cm.discuss_text("I love programming in python")
    status = cm.get_status()
    goal = cm.set_goal("learn more python", priority=0.8)
    cm.pursue_goal(goal.goal_id)
"""

import json
import os
import re
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from .cell_memory import CellMemory, Cell, GlobalPool, HEBB_STRENGTH, INITIAL_STRENGTH


# ============================================================
# Constants
# ============================================================

WM_CAPACITY = 5
WM_DECAY_RATE = 0.85
WM_REFRESH_BOOST = 0.3
WM_RETRIEVAL_BOOST = 0.4
EMOTION_DECAY_RATE = 0.92
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5
TOOL_RESULT_BOOST = 0.25
DEFAULT_PRIORITY = 0.5
STATE_DIR = os.path.join(os.path.expanduser("~"), ".cellmind")
MAX_HISTORY = 50

os.makedirs(STATE_DIR, exist_ok=True)

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both",
    "not", "only", "own", "same", "than", "too", "very", "just",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "how", "when", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such",
    "no", "any", "say", "said", "get", "got", "make", "made",
    "go", "went", "come", "came", "take", "took", "see", "saw",
    "know", "knew", "think", "thought", "give", "gave", "tell", "told",
    "ask", "asked", "use", "used", "find", "found", "want", "wanted",
    "because", "also", "now", "here", "there", "up", "down", "out",
    "about", "over", "back", "well", "way", "thing", "things",
    "something", "nothing", "anything", "everything", "one", "ones",
    "two", "three", "first", "second", "new", "old", "good", "bad",
    "great", "right", "left", "much", "many", "even", "still", "never",
    "always", "ever", "however", "though", "although", "while",
    "please", "thanks", "thank", "sorry", "yes", "no", "ok", "okay",
    "hello", "hi", "hey", "bye", "goodbye", "like", "just", "really",
}

EMOTION_BOOST = {
    "happy": 1.2, "excited": 1.3, "neutral": 1.0, "sad": 0.85,
    "angry": 1.15, "fearful": 0.8, "frustrated": 0.9,
}

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "happy", "joy", "excited", "brilliant", "perfect", "awesome",
    "beautiful", "delightful", "thrilled", "glad", "pleased", "successful",
    "helpful", "useful", "good", "best", "better",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "sad", "angry",
    "frustrated", "disappointed", "confused", "wrong", "fail", "failed",
    "error", "problem", "issue", "difficult", "hard", "confusing",
    "stuck", "annoying", "poor", "worst", "useless", "broken",
}


# ============================================================
# Token extraction
# ============================================================

def extract_tokens(text: str, min_len: int = 3) -> List[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOP_WORDS]


def extract_key_phrases(text: str) -> List[str]:
    return extract_tokens(text, min_len=3)[:12]


# ============================================================
# Emotion
# ============================================================

@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    emotion_label: str = "neutral"

    def classify(self) -> str:
        v, a, d = self.valence, self.arousal, self.dominance
        if v >= POSITIVE_THRESHOLD:
            label = "excited" if a >= 0.7 else "happy"
        elif v <= NEGATIVE_THRESHOLD:
            label = "angry" if (a >= 0.6 and d >= 0.5) else "fearful" if a >= 0.6 else "sad"
        else:
            label = "frustrated" if (a >= 0.7 and d >= 0.6) else "neutral"
        self.emotion_label = label
        return label

    def get_boost(self) -> float:
        return EMOTION_BOOST.get(self.classify(), 1.0)

    def apply_sentiment(self, delta: float):
        self.valence = max(-1.0, min(1.0, self.valence + delta * 0.5))
        if abs(delta) > 0.5:
            self.arousal = max(0.0, min(1.0, self.arousal + delta * 0.2))

    def drift(self):
        self.valence *= EMOTION_DECAY_RATE
        self.arousal = 0.5 + (self.arousal - 0.5) * 0.9
        self.classify()

    def status(self) -> str:
        return f"{self.classify()}(v={self.valence:+.2f}, a={self.arousal:.2f})"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "EmotionState":
        return EmotionState(
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.5),
            dominance=d.get("dominance", 0.5),
            emotion_label=d.get("emotion_label", "neutral"),
        )


def detect_sentiment(text: str) -> Tuple[float, str]:
    all_words = set(re.findall(r"[a-z]+", text.lower()))
    words = {w for w in all_words if w not in STOP_WORDS or w in POSITIVE_WORDS or w in NEGATIVE_WORDS}
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0, "neutral"
    base = max(-1.0, min(1.0, (pos - neg) / total))
    label = "positive" if base > 0.5 else "negative" if base < -0.5 else "neutral"
    return base, label


# ============================================================
# Working Memory
# ============================================================

@dataclass
class WorkingCell:
    ltm_cell_id: str
    preference: str
    wm_activation: float = 1.0
    retrieval_count: int = 0
    last_accessed: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "WorkingCell":
        return WorkingCell(
            ltm_cell_id=d["ltm_cell_id"],
            preference=d["preference"],
            wm_activation=d.get("wm_activation", 1.0),
            retrieval_count=d.get("retrieval_count", 0),
            last_accessed=d.get("last_accessed", ""),
        )


class WorkingMemory:
    """Short-term memory with limited capacity (5 items by default)."""

    def __init__(self, cell_memory: CellMemory, capacity: int = WM_CAPACITY):
        self.cell_memory = cell_memory
        self.capacity = capacity
        self.items: Dict[str, WorkingCell] = {}
        self.total_retrievals: int = 0

    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def retrieve(self, preference: str) -> Optional[WorkingCell]:
        candidates = [
            c for c in self.cell_memory.cells.values()
            if c.strength > 0.1 and (preference in c.preference or c.preference in preference)
        ]
        if not candidates:
            candidates = list(self.cell_memory.cells.values())[:5]
        scored = [
            (len(set(c.preference) & set(preference)) / max(len(c.preference), 1), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: (x[0], x[1].strength), reverse=True)
        best = scored[0][1] if scored else None
        if not best:
            return None
        key = best.preference
        existing = self.items.get(key)
        if existing:
            existing.wm_activation = min(1.5, existing.wm_activation + WM_REFRESH_BOOST)
            existing.retrieval_count += 1
            existing.last_accessed = datetime.now().isoformat()
            return existing
        if self.is_full():
            weakest = min(self.items, key=lambda k: self.items[k].wm_activation)
            del self.items[weakest]
        wc = WorkingCell(
            ltm_cell_id=best.cell_id,
            preference=best.preference,
            wm_activation=1.0 + WM_RETRIEVAL_BOOST,
            retrieval_count=1,
            last_accessed=datetime.now().isoformat(),
        )
        self.items[key] = wc
        self.total_retrievals += 1
        return wc

    def decay(self):
        decayed = [k for k, wc in self.items.items() if wc.wm_activation < 0.1]
        for k in decayed:
            del self.items[k]
        for wc in self.items.values():
            wc.wm_activation *= WM_DECAY_RATE

    def get_focus(self) -> List[WorkingCell]:
        return sorted(self.items.values(), key=lambda wc: wc.wm_activation, reverse=True)

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "decay_rate": WM_DECAY_RATE,
            "total_retrievals": self.total_retrievals,
            "items": {k: v.to_dict() for k, v in self.items.items()},
        }

    @staticmethod
    def from_dict(data: dict, cell_memory: CellMemory) -> "WorkingMemory":
        wm = WorkingMemory(cell_memory, capacity=data.get("capacity", WM_CAPACITY))
        wm.total_retrievals = data.get("total_retrievals", 0)
        for pref, wc_data in data.get("items", {}).items():
            wc = WorkingCell.from_dict(wc_data)
            found = None
            for cell in cell_memory.cells.values():
                if cell.preference == pref:
                    found = cell
                    break
            if found:
                wc.ltm_cell_id = found.cell_id
                if wc.last_accessed:
                    try:
                        elapsed = (datetime.now() - datetime.fromisoformat(wc.last_accessed)).total_seconds()
                        n = int(min(elapsed, 3600))
                        for _ in range(max(1, n)):
                            wc.wm_activation *= WM_DECAY_RATE
                    except (ValueError, TypeError):
                        pass
                if wc.wm_activation >= 0.1:
                    wm.items[pref] = wc
        return wm


# ============================================================
# Goals
# ============================================================

@dataclass
class GoalStep:
    step_id: str
    description: str
    topic: str
    state: str = "pending"
    attempts: int = 0
    result: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GoalStep":
        return GoalStep(**d)


@dataclass
class Goal:
    goal_id: str
    description: str
    priority: float = DEFAULT_PRIORITY
    state: str = "pending"
    progress: float = 0.0
    steps: List[GoalStep] = field(default_factory=list)
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def completion_ratio(self) -> float:
        if not self.steps:
            return self.progress
        return sum(1 for s in self.steps if s.state == "completed") / len(self.steps)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Goal":
        d = dict(d)
        d["steps"] = [GoalStep.from_dict(s) for s in d.get("steps", [])]
        return Goal(**d)


# ============================================================
# CellMindCore
# ============================================================

class CellMindCore:
    """
    Full CellMind integration: memory + emotion + goals + working memory.

    Usage:
        cm = CellMindCore()
        cm.discuss_text("programming in python")
        status = cm.get_status()
        goal = cm.set_goal("master python", priority=0.8)
        cm.pursue_goal(goal.goal_id)

    Attributes:
        cell_memory: CellMemory — long-term cell memory
        working_memory: WorkingMemory — short-term focus
        emotion: EmotionState — current emotional state
        goals: List[Goal] — active goals
    """

    def __init__(self, state_dir: str = STATE_DIR):
        self.state_dir = state_dir
        self.cell_memory = CellMemory()
        self.working_memory = WorkingMemory(self.cell_memory)
        self.emotion = EmotionState()
        self.goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.conversation_history: List[dict] = []
        self._state_file = os.path.join(state_dir, "cellmind_state.json")
        self._lock = threading.RLock()
        self._load()

    # ---- Core operations ----

    def activate_text(
        self,
        text: str,
        goal_id: Optional[str] = None,
        boost_override: Optional[float] = None,
    ) -> List[str]:
        tokens = extract_key_phrases(text)
        boost = boost_override if boost_override is not None else HEBB_STRENGTH * self.emotion.get_boost()
        with self._lock:
            if goal_id:
                self._activate_goal_tokens(tokens, goal_id, boost)
            else:
                self.cell_memory.activate_tokens(tokens)
            for t in tokens:
                self.working_memory.retrieve(t)
        return tokens

    def _activate_goal_tokens(self, tokens: List[str], goal_id: str, boost: float):
        prev = None
        for token in tokens:
            cell = self.cell_memory.activate(token)
            if goal_id not in cell.goal_tags:
                cell.goal_tags.append(goal_id)
            if prev and prev in self.cell_memory.cells and cell.cell_id in self.cell_memory.cells:
                w = cell.connections.get(prev, 0.0)
                cell.connections[prev] = min(1.0, w + HEBB_STRENGTH)
            prev = cell.cell_id

    def discuss_text(self, text: str) -> dict:
        """
        Process a discussion turn: token extraction, cell activation,
        sentiment detection, and emotion update.

        Args:
            text: Discussion text

        Returns:
            dict with tokens_activated, sentiment, emotion, wm_focus, top_cells
        """
        tokens = self.activate_text(text)
        sentiment_delta, sentiment_label = detect_sentiment(text)
        with self._lock:
            self.emotion.apply_sentiment(sentiment_delta)
            self.emotion.drift()
            self.conversation_history.append({
                "text": text,
                "tokens": tokens,
                "sentiment": sentiment_delta,
                "emotion": self.emotion.valence,
            })
        return {
            "tokens_activated": tokens,
            "sentiment": sentiment_label,
            "emotion": self.emotion.status(),
            "wm_focus": [wc.preference for wc in self.working_memory.get_focus()],
            "top_cells": [c.preference for c in self.cell_memory.get_top_cells(5)],
        }

    def set_goal(self, description: str, priority: float = DEFAULT_PRIORITY) -> Goal:
        """Create a new goal with token-based steps."""
        goal_id = f"goal_{len(self.goals) + len(self.completed_goals) + 1}"
        goal = Goal(goal_id=goal_id, description=description, priority=priority)
        tokens = extract_key_phrases(description)
        for i, t in enumerate(tokens[:4]):
            goal.steps.append(GoalStep(
                step_id=f"step_{i+1}",
                description=f"Explore {t}",
                topic=t,
            ))
        with self._lock:
            self.goals.append(goal)
            self.save()
        return goal

    def pursue_goal(self, goal_id: str) -> dict:
        """Execute all pending steps in a goal."""
        with self._lock:
            goal = next((g for g in self.goals if g.goal_id == goal_id), None)
            if not goal:
                return {"error": f"Goal {goal_id} not found"}
            goal.state = "active"
            results = []
            for step in goal.steps:
                if step.state == "pending":
                    step.state = "active"
                    step.attempts += 1
                    self.activate_text(step.topic, goal_id=goal_id)
                    step.state = "completed"
                    step.result = f"Explored: {step.topic}"
                    results.append(step.result)
            goal.progress = goal.completion_ratio()
            if goal.progress >= 1.0:
                goal.state = "completed"
                goal.completed_at = datetime.now().isoformat()
                self.completed_goals.append(goal)
                self.goals.remove(goal)
            self.save()
        return {"goal_id": goal_id, "progress": goal.progress, "results": results}

    # ---- Persistence ----

    def save(self):
        """Save full state to disk."""
        data = {
            "cell_memory": self.cell_memory.to_dict(),
            "working_memory": self.working_memory.to_dict(),
            "emotion": self.emotion.to_dict(),
            "goals": [g.to_dict() for g in self.goals],
            "completed_goals": [g.to_dict() for g in self.completed_goals],
            "conversation_history": self.conversation_history,
            "_saved_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cell_memory = CellMemory.from_dict(data.get("cell_memory", {}))
            self.working_memory = WorkingMemory.from_dict(
                data.get("working_memory", {}), self.cell_memory)
            self.emotion = EmotionState.from_dict(data.get("emotion", {}))
            self.goals = [Goal.from_dict(g) for g in data.get("goals", [])]
            self.completed_goals = [Goal.from_dict(g) for g in data.get("completed_goals", [])]
            self.conversation_history = data.get("conversation_history", [])
        except Exception:
            pass

    # ---- State queries ----

    def get_status(self) -> dict:
        """Return full system status."""
        top = self.cell_memory.get_top_cells(8)
        wm_focus = self.working_memory.get_focus()
        return {
            "cells_total": len(self.cell_memory.cells),
            "top_cells": [
                {"preference": c.preference, "strength": round(c.strength, 3),
                 "connections": len(c.connections), "goal_tags": c.goal_tags}
                for c in top
            ],
            "wm_items": [
                {"preference": wc.preference, "activation": round(wc.wm_activation, 3),
                 "retrievals": wc.retrieval_count}
                for wc in wm_focus
            ],
            "wm_capacity": self.working_memory.capacity,
            "global_pool": round(self.cell_memory.global_pool.energy, 3),
            "emotion": self.emotion.status(),
            "emotion_valence": round(self.emotion.valence, 3),
            "active_goals": [
                {"id": g.goal_id, "description": g.description,
                 "priority": g.priority, "progress": round(g.completion_ratio(), 2),
                 "state": g.state}
                for g in self.goals
            ],
            "completed_goals": [
                {"id": g.goal_id, "description": g.description}
                for g in self.completed_goals[-5:]
            ],
            "conversation_count": len(self.conversation_history),
        }

    def get_context_prompt(self) -> str:
        """Generate system prompt injection for LLM context."""
        status = self.get_status()
        top_prefs = [c["preference"] for c in status["top_cells"]]
        wm_prefs = [w["preference"] for w in status["wm_items"]]
        active_goals = status["active_goals"]

        lines = [
            "",
            "=== CellMind Memory State ===",
            f"Cells: {len(self.cell_memory.cells)} total | Top: {', '.join(top_prefs[:6]) if top_prefs else 'none'}",
            f"Working Memory focus: {', '.join(wm_prefs) if wm_prefs else 'none'} / {self.working_memory.capacity}",
            f"Emotion: {status['emotion']} | Conversations: {status['conversation_count']}",
        ]
        if active_goals:
            lines.append(f"Active Goals: {', '.join(g['description'] for g in active_goals)}")
        if status["completed_goals"]:
            lines.append(f"Recent Completed: {', '.join(g['description'] for g in status['completed_goals'][-3:])}")
        lines.append("=== End CellMind Memory ===")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return self.get_status()

    def __repr__(self):
        return f"CellMindCore(cells={len(self.cell_memory.cells)}, goals={len(self.goals)})"


# ---- Cell extension ----
Cell.goal_tags = field(default_factory=list)