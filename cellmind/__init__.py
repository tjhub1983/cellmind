# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
CellMind: biological cell-based AI memory architecture.

Installation:
    pip install cellmind

Usage:
    from cellmind import CellMindCore

    cm = CellMindCore()
    cm.discuss_text("CellMind solves AI memory continuity")
    status = cm.get_status()
"""

from .core import (
    CellMindCore,
    WorkingMemory,
    EmotionState,
    Goal,
    GoalStep,
    detect_sentiment,
    extract_tokens,
    extract_key_phrases,
)
from .cell_memory import CellMemory, Cell, GlobalPool
from .sc_engine import SCEngine, REMWrapper

__version__ = "0.1.0"

__all__ = [
    "CellMindCore",
    "CellMemory",
    "Cell",
    "GlobalPool",
    "WorkingMemory",
    "EmotionState",
    "Goal",
    "GoalStep",
    "SCEngine",
    "REMWrapper",
    "detect_sentiment",
    "extract_tokens",
    "extract_key_phrases",
]