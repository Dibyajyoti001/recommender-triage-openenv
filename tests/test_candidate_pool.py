import numpy as np

from app.candidate_pool import build_candidate_pool


def test_candidate_pool_has_exact_six_slots():
    rng = np.random.default_rng(123)
    items = build_candidate_pool(
        task_id="task_1",
        turn=0,
        alpha=0.4,
        eta_q=0.25,
        z=[0.5, 0.2, 0.1, 0.1, 0.1],
        m=[0.5, 0.2, 0.1, 0.1, 0.1],
        item_fatigue={},
        history_categories=[],
        rng=rng,
    )
    assert len(items) == 6
    slot_types = {item.slot_type for item in items}
    assert slot_types == {
        "live_best_fresh",
        "live_best_fatigued",
        "memory_best_fresh",
        "plausible_distractor",
        "novel_risky_distractor",
        "neutral_filler",
    }


def test_task3_forces_conflict():
    rng = np.random.default_rng(456)
    items = build_candidate_pool(
        task_id="task_3",
        turn=0,
        alpha=0.8,
        eta_q=0.25,
        z=[0.5, 0.2, 0.1, 0.1, 0.1],
        m=[0.5, 0.2, 0.1, 0.1, 0.1],
        item_fatigue={},
        history_categories=[],
        rng=rng,
    )
    live_item = next(i for i in items if i.slot_type == "live_best_fresh")
    mem_item = next(i for i in items if i.slot_type == "memory_best_fresh")
    assert live_item.category_id != mem_item.category_id