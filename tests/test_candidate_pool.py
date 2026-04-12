from __future__ import annotations

import numpy as np

from app.candidate_pool import build_candidate_pool
from app.tasks import CATEGORY_NAMES, K


def _is_prob_vector(v: list[float], tol: float = 1e-6) -> bool:
    return len(v) == K and all(x >= -tol for x in v) and abs(sum(v) - 1.0) <= tol


def test_candidate_pool_has_exact_six_slots():
    rng = np.random.default_rng(123)

    z = [0.36, 0.22, 0.14, 0.06, 0.03, 0.08, 0.03, 0.02, 0.03, 0.03]
    m = [0.32, 0.24, 0.16, 0.06, 0.03, 0.07, 0.03, 0.02, 0.03, 0.04]

    items = build_candidate_pool(
        task_id="task_1",
        turn=0,
        alpha=0.45,
        eta_q=0.25,
        z=z,
        m=m,
        item_fatigue={},
        history_categories=[],
        rng=rng,
    )

    assert len(items) == 6

    slot_types = {item.slot_type for item in items}
    assert slot_types == {
        "live_best_fresh",
        "memory_best_fresh",
        "balanced_bridge",
        "fatigue_trap",
        "exploration_option",
        "conflict_option",
    }

    for item in items:
        assert 0 <= item.category_id < K
        assert item.category_name == CATEGORY_NAMES[item.category_id]
        assert _is_prob_vector(item.topic_vector)
        assert 0.0 <= item.quality <= 1.0
        assert 0.0 <= item.engagement <= 1.0
        assert 0.0 <= item.cost <= 1.0
        assert 0.0 <= item.risk <= 1.0
        assert 0.0 <= item.latency <= 1.0
        assert item.freshness in {"fresh", "stale", "novel"}


def test_task3_conflict_option_points_away_from_memory():
    rng = np.random.default_rng(456)

    # Strong memory in documentary/crime/news region
    m = [0.04, 0.06, 0.08, 0.10, 0.04, 0.26, 0.18, 0.04, 0.14, 0.06]
    # Live in romance/comedy/drama region
    z = [0.28, 0.30, 0.18, 0.06, 0.02, 0.04, 0.02, 0.02, 0.04, 0.04]

    items = build_candidate_pool(
        task_id="task_3",
        turn=0,
        alpha=0.82,
        eta_q=0.25,
        z=z,
        m=m,
        item_fatigue={},
        history_categories=[],
        rng=rng,
    )

    by_slot = {item.slot_type: item for item in items}

    conflict = by_slot["conflict_option"]
    memory_best = by_slot["memory_best_fresh"]
    live_best = by_slot["live_best_fresh"]

    dot_conflict_memory = sum(a * b for a, b in zip(conflict.topic_vector, m))
    dot_memorybest_memory = sum(a * b for a, b in zip(memory_best.topic_vector, m))
    dot_livebest_live = sum(a * b for a, b in zip(live_best.topic_vector, z))

    # conflict option should be less aligned with memory than memory_best_fresh
    assert dot_conflict_memory < dot_memorybest_memory

    # live_best_fresh should still track live intent reasonably well
    assert dot_livebest_live > 0.15

    # all items remain valid sparse-ish distributions
    for item in items:
        assert _is_prob_vector(item.topic_vector)


def test_distressed_regime_opens_safer_recovery_lanes():
    z = [0.02, 0.03, 0.04, 0.04, 0.03, 0.08, 0.04, 0.07, 0.58, 0.07]
    m = [0.02, 0.03, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.55, 0.07]

    stable_items = build_candidate_pool(
        task_id="task_4",
        turn=10,
        alpha=0.52,
        eta_q=0.25,
        z=z,
        m=m,
        item_fatigue={},
        history_categories=[8, 8, 8, 5],
        rng=np.random.default_rng(999),
        regime=0,
        latent_vol=0.12,
    )
    distressed_items = build_candidate_pool(
        task_id="task_4",
        turn=10,
        alpha=0.52,
        eta_q=0.25,
        z=z,
        m=m,
        item_fatigue={},
        history_categories=[8, 8, 8, 5],
        rng=np.random.default_rng(999),
        regime=3,
        latent_vol=0.70,
    )

    stable_by_slot = {item.slot_type: item for item in stable_items}
    distressed_by_slot = {item.slot_type: item for item in distressed_items}

    assert distressed_by_slot["exploration_option"].risk < stable_by_slot["exploration_option"].risk
    assert distressed_by_slot["balanced_bridge"].cost < stable_by_slot["balanced_bridge"].cost
    assert distressed_by_slot["fatigue_trap"].risk > stable_by_slot["fatigue_trap"].risk
