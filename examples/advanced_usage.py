# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
Advanced usage: SC Engine, REM, Goals, and persistence.

Run:
    python examples/advanced_usage.py
"""

from cellmind import CellMindCore, SCEngine, REMWrapper


def sc_engine_demo():
    print("=== SC Engine Demo ===")
    sc = SCEngine()

    print("Injecting threat signal...")
    sc.receive_signal({"pheromone": "threat", "intensity": 0.8, "metadata": {"severity": 0.7}})
    print(f"  Fear level: {sc.get_fear_level():.3f}")

    print("\nInjecting reward signal...")
    sc.receive_signal({"pheromone": "dopamine", "intensity": 0.9})
    print(f"  Dopamine level: {sc.get_dopamine_level():.3f}")

    print("\nAll channels:")
    status = sc.get_status()
    for ch, val in status["channels"].items():
        print(f"  {ch}: {val:.3f}")

    print("\nTick (regulation decay)...")
    result = sc.tick()
    print(f"  sc_tick: {result['sc_tick']}")
    print()


def rem_demo():
    print("=== REM Fragment Demo ===")
    rem = REMWrapper()

    print("Adding memory fragments...")
    frag1 = rem.add_memory(
        "discussed python cell memory architecture",
        importance=0.9,
        tags=["python", "memory"]
    )
    frag2 = rem.add_memory(
        "learned about Hebbian learning",
        importance=0.7,
        tags=["learning", "neural"]
    )
    frag3 = rem.add_memory(
        "CellMind solves context loss",
        importance=0.95,
        tags=["cellmind", "problem"]
    )
    print(f"  Added fragments: {frag1}, {frag2}, {frag3}")

    print("\nSearching for 'memory'...")
    results = rem.search("memory")
    for r in results:
        print(f"  [{r['importance']:.2f}] {r['content']}")

    print("\nSearching by tag 'python'...")
    results = rem.search(tag="python")
    for r in results:
        print(f"  [{r['importance']:.2f}] {r['content']}")

    print(f"\nStats: {rem.get_stats()}")
    print(f"Vitality: {rem.get_vitality():.3f}")
    print()


def goal_system_demo():
    print("=== Goal System Demo ===")
    cm = CellMindCore()

    print("Creating a multi-step goal...")
    goal = cm.set_goal("build a memory-powered app", priority=0.9)
    print(f"  Goal: {goal.goal_id} - {goal.description}")
    print(f"  Priority: {goal.priority}")
    print(f"  Steps: {len(goal.steps)}")
    for s in goal.steps:
        print(f"    - {s.step_id}: {s.topic}")

    print("\nPursuing goal...")
    result = cm.pursue_goal(goal.goal_id)
    print(f"  Progress: {result['progress']:.0%}")
    print(f"  Results: {result['results']}")

    status = cm.get_status()
    print(f"\nCompleted goals: {[g['description'] for g in status['completed_goals']]}")
    print()


def persistence_demo():
    print("=== Persistence Demo ===")
    import tempfile, os, shutil

    state_dir = tempfile.mkdtemp()
    print(f"Using temp dir: {state_dir}")

    print("\nCreating CellMind and adding data...")
    cm = CellMindCore(state_dir=state_dir)
    cm.discuss_text("python memory system design")
    cm.discuss_text("CellMind biological cell model")
    cm.set_goal("test persistence", priority=0.8)
    print(f"  Cells: {len(cm.cell_memory.cells)}")
    print(f"  Goals: {len(cm.goals)}")
    cm.save()

    print("\nReinitializing CellMind from saved state...")
    cm2 = CellMindCore(state_dir=state_dir)
    print(f"  Cells: {len(cm2.cell_memory.cells)}")
    print(f"  Goals: {len(cm2.goals)}")
    print(f"  Conversations: {cm2.get_status()['conversation_count']}")

    shutil.rmtree(state_dir)
    print("\nTemp dir cleaned up.")
    print()


if __name__ == "__main__":
    sc_engine_demo()
    rem_demo()
    goal_system_demo()
    persistence_demo()
    print("All demos complete.")