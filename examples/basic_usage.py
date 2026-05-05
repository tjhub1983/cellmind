# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
Basic usage example for CellMind.

Run:
    python examples/basic_usage.py
"""

from cellmind import CellMindCore, CellMemory


def basic_cell_memory():
    print("=== CellMemory Demo ===")
    cm = CellMemory()

    print(f"Initial: {len(cm.cells)} cells")
    cm.activate("python")
    cm.activate("python")
    cm.activate("programming")
    print(f"After activation: {len(cm.cells)} cells")

    top = cm.get_top_cells(5)
    print("Top cells:")
    for c in top:
        print(f"  {c.preference}: strength={c.strength:.2f}, responses={c.response_count}")

    print(f"\nGlobalPool: {cm.global_pool.energy:.3f}")
    cm.decay_all(days=7)
    print(f"After 7-day decay: {len(cm.cells)} cells")
    top2 = cm.get_top_cells(5)
    for c in top2:
        print(f"  {c.preference}: strength={c.strength:.2f}")
    print()


def basic_cmind_core():
    print("=== CellMindCore Demo ===")
    cm = CellMindCore()

    print("Discussing topics...")
    cm.discuss_text("I love programming in python")
    cm.discuss_text("Python has great memory systems")
    cm.discuss_text("CellMind solves AI memory continuity")

    status = cm.get_status()
    print(f"\nCells: {status['cells_total']}")
    print(f"Top concepts: {[c['preference'] for c in status['top_cells'][:5]]}")
    print(f"Working Memory: {[w['preference'] for w in status['wm_items']]}")
    print(f"Emotion: {status['emotion']}")
    print(f"Conversations: {status['conversation_count']}")
    print()

    print("Setting a goal...")
    goal = cm.set_goal("learn python memory architecture", priority=0.85)
    print(f"  Created: {goal.goal_id} - {goal.description}")
    print(f"  Steps: {[s.topic for s in goal.steps]}")

    print("\nPursuing goal...")
    result = cm.pursue_goal(goal.goal_id)
    print(f"  Progress: {result['progress']:.0%}")
    print(f"  Results: {result['results']}")
    print()

    print("Context prompt for LLM injection:")
    print(cm.get_context_prompt())


if __name__ == "__main__":
    basic_cell_memory()
    print()
    basic_cmind_core()
    print("\nDone.")