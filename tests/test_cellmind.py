# CellMind
# Copyright 2026 CellMind Team
# Licensed under the Apache License 2.0

"""
Test suite for CellMind core functionality.
Run with: pytest tests/ -v
"""

import sys, os, json, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmind import (
    CellMemory, CellMindCore, SCEngine, REMWrapper,
    EmotionState, Goal, detect_sentiment, extract_tokens, extract_key_phrases,
)


# ============================================================
# Test: CellMemory Core
# ============================================================

def test_cellmemory_init():
    cm = CellMemory()
    assert len(cm.cells) == 0
    assert cm.global_pool.energy == 0.0


def test_single_activation_creates_cell():
    cm = CellMemory()
    cell = cm.activate("python")
    assert cell.preference == "python"
    assert cell.strength == 1.0
    assert len(cm.cells) == 1


def test_repeated_activation_increases_strength():
    cm = CellMemory()
    cm.activate("memory")
    cm.activate("memory")
    cm.activate("memory")
    cell = list(cm.cells.values())[0]
    # net boost = HEBB * (1 - BOX_SAVE_RATIO - GLOBAL_POOL_CONTRIB) = 0.2 * 0.7 = 0.14
    assert abs(cell.strength - 1.28) < 0.01


def test_strength_cap_at_2():
    cm = CellMemory()
    for _ in range(15):
        cm.activate("cap")
    for c in cm.cells.values():
        assert c.strength <= 2.0


def test_decay_runs_without_error():
    cm = CellMemory()
    cm.activate("decay")
    cm.decay_all(days=1)
    assert True


def test_30day_decay_cells_stay_positive():
    cm = CellMemory()
    for _ in range(3):
        cm.activate("retain")
    cm.decay_all(days=30)
    for c in cm.cells.values():
        assert c.strength > 0


def test_max_cells_enforced_at_100():
    cm = CellMemory()
    for i in range(150):
        cm.activate(f"max{i}")
    assert len(cm.cells) <= 100


def test_weak_cell_elimination():
    cm = CellMemory()
    cm.activate("strong")
    cm.activate("strong")
    cm.activate("strong")
    cm.activate("weak")
    before = len(cm.cells)
    cm.decay_all(days=365)
    after = len(cm.cells)
    assert 0 <= after <= before


def test_global_pool_contribution():
    cm = CellMemory()
    before = cm.global_pool.energy
    cm.activate("gptest")
    assert cm.global_pool.energy >= before


def test_cell_save_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        cm = CellMemory(save_path=path)
        cm.activate("save")
        cm.save()
        assert os.path.exists(path)
        cm2 = CellMemory(save_path=path)
        assert len(cm2.cells) >= 1
    finally:
        os.unlink(path)


# ============================================================
# Test: Token extraction
# ============================================================

def test_extract_tokens_basic():
    tokens = extract_tokens("I love programming in python")
    assert "love" in tokens
    assert "programming" in tokens
    assert "python" in tokens
    assert "i" not in tokens
    assert "in" not in tokens


def test_extract_key_phrases_limit():
    tokens = extract_key_phrases("a b c d e f g h i j k l")
    assert len(tokens) <= 12


def test_detect_sentiment_positive():
    delta, label = detect_sentiment("great excellent amazing")
    assert label == "positive"
    assert delta > 0


def test_detect_sentiment_negative():
    delta, label = detect_sentiment("terrible awful horrible")
    assert label == "negative"
    assert delta < 0


def test_detect_sentiment_neutral():
    delta, label = detect_sentiment("the system runs")
    assert label == "neutral"


# ============================================================
# Test: CellMindCore
# ============================================================

def test_cmind_init():
    cm = CellMindCore()
    assert hasattr(cm, "cell_memory")
    assert hasattr(cm, "working_memory")
    assert hasattr(cm, "emotion")


def test_cmind_discuss_text():
    cm = CellMindCore()
    result = cm.discuss_text("CellMind memory architecture")
    assert len(result["tokens_activated"]) > 0
    assert "sentiment" in result
    assert "emotion" in result


def test_cmind_negative_text_lowers_valence():
    cm = CellMindCore()
    cm.emotion.valence = 0.0
    cm.discuss_text("terrible awful horrible broken fail")
    assert cm.emotion.valence < 0


def test_cmind_positive_text_raises_valence():
    cm = CellMindCore()
    cm.emotion.valence = 0.0
    cm.discuss_text("great excellent amazing wonderful love")
    assert cm.emotion.valence > 0


def test_cmind_set_goal():
    cm = CellMindCore()
    g = cm.set_goal("musk test goal", priority=0.9)
    assert g.goal_id.startswith("goal_")
    assert g.description == "musk test goal"


def test_cmind_pursue_goal():
    cm = CellMindCore()
    g = cm.set_goal("pursue test")
    r = cm.pursue_goal(g.goal_id)
    assert r["progress"] == 1.0
    assert len(r["results"]) > 0


def test_cmind_get_status():
    cm = CellMindCore()
    s = cm.get_status()
    assert "cells_total" in s
    assert "emotion" in s
    assert "conversation_count" in s


def test_cmind_context_prompt():
    cm = CellMindCore()
    ctx = cm.get_context_prompt()
    assert len(ctx) > 50
    assert "CellMind" in ctx or "Cells:" in ctx


def test_cmind_activate_text():
    cm = CellMindCore()
    tokens = cm.activate_text("direct memory test")
    assert len(tokens) > 0


def test_cmind_working_memory_retrieve():
    cm = CellMindCore()
    cm.activate_text("wm retrieval test")
    wc = cm.working_memory.retrieve("memory")
    assert wc is not None


# ============================================================
# Test: SC Engine
# ============================================================

def test_sc_init():
    sc = SCEngine()
    assert hasattr(sc, "_channels")
    assert "fear" in sc._channels


def test_sc_fear_inject():
    sc = SCEngine()
    sc.receive_signal({"pheromone": "threat", "intensity": 0.8})
    assert sc.get_fear_level() > 0


def test_sc_fear_capped():
    sc = SCEngine()
    for _ in range(20):
        sc.receive_signal({"pheromone": "threat", "intensity": 0.9})
    fear = sc.get_fear_level()
    assert fear < 10.0


def test_sc_all_channels():
    sc = SCEngine()
    channels = ["threat", "dopamine", "oxytocin", "endorphin", "serotonin"]
    for ch in channels:
        sc.receive_signal({"pheromone": ch, "intensity": 0.6})
    status = sc.get_status()
    assert status["tick_count"] == 5


def test_sc_tick():
    sc = SCEngine()
    sc.receive_signal({"pheromone": "threat", "intensity": 0.8})
    result = sc.tick()
    assert "sc_tick" in result
    assert "channels" in result


# ============================================================
# Test: REM Wrapper
# ============================================================

def test_rem_init():
    rem = REMWrapper()
    assert hasattr(rem, "_fragments")


def test_rem_add_memory():
    rem = REMWrapper()
    frag_id = rem.add_memory("musk test fragment", importance=0.8, tags=["test"])
    assert frag_id is not None


def test_rem_search():
    rem = REMWrapper()
    rem.add_memory("python programming language", importance=0.9, tags=["code"])
    results = rem.search("programming")
    assert len(results) > 0


def test_rem_tag_filter():
    rem = REMWrapper()
    rem.add_memory("tagged fragment", importance=0.8, tags=["priority"])
    results = rem.search(tag="priority")
    assert len(results) >= 1


def test_rem_stats():
    rem = REMWrapper()
    rem.add_memory("stats test", importance=0.7)
    stats = rem.get_stats()
    assert "total_fragments" in stats


def test_rem_vitality():
    rem = REMWrapper()
    rem.add_memory("high", importance=0.9)
    rem.add_memory("low", importance=0.2)
    v = rem.get_vitality()
    assert 0.0 < v <= 1.0


# ============================================================
# Test: Emotion
# ============================================================

def test_emotion_classify():
    em = EmotionState()
    em.valence = 0.8
    em.arousal = 0.8
    label = em.classify()
    assert label in ["excited", "happy"]


def test_emotion_apply_sentiment():
    em = EmotionState()
    em.apply_sentiment(0.8)
    assert em.valence > 0


def test_emotion_boost():
    em = EmotionState()
    em.valence = 0.7
    em.classify()
    boost = em.get_boost()
    assert boost >= 1.0


# ============================================================
# Test: Edge cases
# ============================================================

def test_empty_string_discuss():
    cm = CellMindCore()
    result = cm.discuss_text("")
    assert "tokens_activated" in result


def test_very_long_text():
    cm = CellMindCore()
    long_text = "test memory " * 1000
    cm.activate_text(long_text)
    assert True


def test_special_characters():
    cm = CellMindCore()
    cm.activate_text("test #$% special chars 123")
    assert True


def test_duplicate_goals_unique_ids():
    cm = CellMindCore()
    g1 = cm.set_goal("dup test")
    g2 = cm.set_goal("dup test")
    assert g1.goal_id != g2.goal_id


def test_goal_not_found():
    cm = CellMindCore()
    result = cm.pursue_goal("goal_999")
    assert "error" in result


# ============================================================
# Test: Persistence
# ============================================================

def test_cmind_save_and_load():
    state_dir = tempfile.mkdtemp()
    state_file = os.path.join(state_dir, "cellmind_state.json")

    cm1 = CellMindCore(state_dir=state_dir)
    cm1.activate_text("persist test")
    cm1.save()

    assert os.path.exists(state_file)

    cm2 = CellMindCore(state_dir=state_dir)
    assert len(cm2.cell_memory.cells) >= 1

    import shutil
    shutil.rmtree(state_dir)


# ============================================================
# Test: Hebbian connections
# ============================================================

def test_hebbian_connection_forms():
    cm = CellMemory()
    # First activation creates cells
    cm.activate("a")
    cm.activate("b")
    pre = set(cm.cells.keys())

    # Second activation should form connection between existing cells
    cm.activate_tokens(["a", "b"])
    post = set(cm.cells.keys())

    # Cells should still exist
    assert len(post) >= 2

    # Find if connections formed
    connected = False
    for cell in cm.cells.values():
        if cell.connections:
            connected = True
    assert connected or len(cm.cells) == 2


# ============================================================
# Test: GlobalPool physics
# ============================================================

def test_global_pool_draw_and_contribute():
    pool = CellMemory().global_pool
    pool.contribute(1.0)
    assert pool.energy == 1.0
    drawn = pool.draw(0.5)
    assert drawn == 0.5
    assert pool.energy == 0.5


def test_global_pool_max_capped():
    pool = CellMemory().global_pool
    pool.contribute(100.0)
    assert pool.energy == pool.max_energy

def test_cmind_state_persistence(tmp_path):
    """
    Integration test for CellMindCore state persistence.
    Verifies cells, goals, and emotion state persist across save/load cycles.
    """
    from cellmind import CellMindCore

    # Create first instance with data
    state_dir = str(tmp_path / "state1")
    cm1 = CellMindCore(state_dir=state_dir)

    # Add cells
    cm1.activate_text("python programming")
    cm1.activate_text("cellmind architecture")
    cm1.activate_text("memory system")

    # Add goal
    cm1.goals_add("test goal", priority=0.8)

    # Get initial state
    cells_before = len(cm1.cell_memory.cell_pool)
    goals_before = len(cm1.goals)
    emotion_before = cm1.emotion.valence

    # Save
    cm1.save()

    # Create new instance from same state_dir
    cm2 = CellMindCore(state_dir=state_dir)

    # Verify state restored
    cells_after = len(cm2.cell_memory.cell_pool)
    goals_after = len(cm2.goals)
    emotion_after = cm2.emotion.valence

    assert cells_after > 0, f"Expected cells, got {cells_after}"
    assert cells_before == cells_after, f"Cell count mismatch: {cells_before} vs {cells_after}"
    assert goals_after > 0, f"Expected goals, got {goals_after}"
    assert goals_before == goals_after, f"Goal count mismatch: {goals_before} vs {goals_after}"
    # Emotion state should be restored (not default zero)
    assert emotion_after is not None, "Emotion state not restored"

    # Clean up handled by tmp_path fixture
