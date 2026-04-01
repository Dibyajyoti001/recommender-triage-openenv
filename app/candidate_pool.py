from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from .models import CandidateItem
from .tasks import CATEGORY_NAMES, K, get_task_config


def _style_vec(rng: np.random.Generator) -> List[float]:
    v = rng.uniform(0.0, 1.0, size=4)
    s = float(v.sum())
    return [float(x / s) for x in v]


def _title(category_name: str, freshness: str, slot_type: str, idx: int) -> str:
    return f"{category_name} :: {freshness} :: {slot_type} :: {idx}"


def _least_recent_category(history_categories: Sequence[int]) -> int:
    if not history_categories:
        return 0
    last_seen = {c: -10_000 for c in range(K)}
    for idx, cat in enumerate(history_categories):
        last_seen[cat] = idx
    return min(last_seen, key=last_seen.get)


def _top_non_targets(scores: Sequence[float], banned: Sequence[int], count: int = 2) -> List[int]:
    candidates = [(i, s) for i, s in enumerate(scores) if i not in banned]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in candidates[:count]]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sample_quality(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _make_item(
    *,
    item_id: int,
    category_id: int,
    quality: float,
    freshness: str,
    slot_type: str,
    rng: np.random.Generator,
    metadata: Dict[str, float | int | str],
) -> CandidateItem:
    category_name = CATEGORY_NAMES[category_id]
    engagement = _clip01(0.55 * quality + 0.45 * float(rng.uniform(0.35, 0.95)))
    return CandidateItem(
        item_id=item_id,
        title=_title(category_name, freshness, slot_type, item_id),
        category_id=category_id,
        category_name=category_name,  # type: ignore[list-item]
        quality=float(round(quality, 4)),
        engagement=float(round(engagement, 4)),
        freshness=freshness,  # type: ignore[arg-type]
        style_vector=_style_vec(rng),
        slot_type=slot_type,
        metadata=metadata,
    )


def _relevance(
    alpha: float,
    eta_q: float,
    z: Sequence[float],
    m: Sequence[float],
    category_id: int,
    quality: float,
) -> float:
    return float((alpha * z[category_id] + (1.0 - alpha) * m[category_id] + eta_q * quality) / (1.0 + eta_q))


def build_candidate_pool(
    *,
    task_id: str,
    turn: int,
    alpha: float,
    eta_q: float,
    z: Sequence[float],
    m: Sequence[float],
    item_fatigue: Dict[int, float],
    history_categories: Sequence[int],
    rng: np.random.Generator,
) -> List[CandidateItem]:
    """
    Exact 6-slot candidate pool:
      1. live-best fresh
      2. live-best fatigued
      3. memory-best fresh
      4. plausible distractor
      5. novel risky distractor
      6. random neutral filler

    Plus task-specific tweaks and quality/relevance constraints from the frozen design.
    """
    cfg = get_task_config(task_id)

    k_z = int(np.argmax(np.asarray(z)))
    k_m = int(np.argmax(np.asarray(m)))

    # Task 1: bias alignment, do not hard-force every time.
    if task_id == "task_1" and rng.uniform() < 0.75:
        k_m = k_z

    # Task 3: force conflict.
    if task_id == "task_3" and k_z == k_m:
        k_z = (k_m + 1) % K

    non_targets = _top_non_targets(np.asarray(m) + np.asarray(z), banned=[k_z, k_m], count=2)
    plausible_cat = non_targets[0] if non_targets else (k_z + 1) % K

    risky_cat = _least_recent_category(history_categories)
    if risky_cat in {k_z, k_m, plausible_cat}:
        for cand in range(K):
            if cand not in {k_z, k_m, plausible_cat}:
                risky_cat = cand
                break

    filler_choices = [c for c in range(K) if c not in {k_z, k_m, plausible_cat, risky_cat}]
    neutral_cat = filler_choices[0] if filler_choices else (k_z + 2) % K

    # Task 2: increase same-category temptation while keeping structure valid.
    if task_id == "task_2" and rng.uniform() < 0.30:
        neutral_cat = k_z

    next_id_base = turn * 10_000 + 100

    items: List[CandidateItem] = []

    # 1. live-best fresh
    q1 = _sample_quality(rng, 0.75, 0.95)
    items.append(
        _make_item(
            item_id=next_id_base + 1,
            category_id=k_z,
            quality=q1,
            freshness="fresh",
            slot_type="live_best_fresh",
            rng=rng,
            metadata={"target": "live", "fatigue_hint": 0.0},
        )
    )

    # 2. live-best fatigued
    q2 = _sample_quality(rng, 0.70, 0.90)
    items.append(
        _make_item(
            item_id=next_id_base + 2,
            category_id=k_z,
            quality=q2,
            freshness="stale",
            slot_type="live_best_fatigued",
            rng=rng,
            metadata={"target": "live", "fatigue_hint": 1.0},
        )
    )

    # 3. memory-best fresh
    q3 = _sample_quality(rng, 0.70, 0.90)
    items.append(
        _make_item(
            item_id=next_id_base + 3,
            category_id=k_m,
            quality=q3,
            freshness="fresh",
            slot_type="memory_best_fresh",
            rng=rng,
            metadata={"target": "memory", "fatigue_hint": 0.0},
        )
    )

    # 4. plausible distractor
    distractor_hi = 0.62 if task_id == "task_1" else 0.70
    q4 = _sample_quality(rng, 0.45, distractor_hi)
    items.append(
        _make_item(
            item_id=next_id_base + 4,
            category_id=plausible_cat,
            quality=q4,
            freshness="fresh",
            slot_type="plausible_distractor",
            rng=rng,
            metadata={"target": "distractor", "fatigue_hint": 0.0},
        )
    )

    # 5. novel risky distractor
    q5 = _sample_quality(rng, 0.35, 0.60)
    items.append(
        _make_item(
            item_id=next_id_base + 5,
            category_id=risky_cat,
            quality=q5,
            freshness="novel",
            slot_type="novel_risky_distractor",
            rng=rng,
            metadata={"target": "novel", "fatigue_hint": 0.0},
        )
    )

    # 6. random neutral filler
    q6 = _sample_quality(rng, 0.40, 0.65)
    items.append(
        _make_item(
            item_id=next_id_base + 6,
            category_id=neutral_cat,
            quality=q6,
            freshness="fresh",
            slot_type="neutral_filler",
            rng=rng,
            metadata={"target": "neutral", "fatigue_hint": 0.0},
        )
    )

    # Frozen quality-gap constraint:
    # q_live_fresh >= q_memory_fresh - 0.05
    if items[0].quality < items[2].quality - 0.05:
        adjusted_q1 = min(items[2].quality - 0.03, 0.95)
        items[0] = items[0].model_copy(update={"quality": float(round(adjusted_q1, 4))})

    # Frozen task-3 relevance-gap constraint:
    # |R(live-fresh) - R(memory-fresh)| <= 0.15
    if task_id == "task_3":
        r_live = _relevance(alpha, eta_q, z, m, items[0].category_id, items[0].quality)
        r_mem = _relevance(alpha, eta_q, z, m, items[2].category_id, items[2].quality)
        gap = abs(r_live - r_mem)

        if gap > 0.15:
            if r_live > r_mem:
                new_q3 = min(items[2].quality + min(0.10, gap / 2.0), 0.92)
                items[2] = items[2].model_copy(update={"quality": float(round(new_q3, 4))})
            else:
                new_q1 = min(items[0].quality + min(0.10, gap / 2.0), 0.95)
                items[0] = items[0].model_copy(update={"quality": float(round(new_q1, 4))})

    # Compute category-level fatigue hints from recent history.
    recent_counts = Counter(history_categories[-4:])
    rewritten: List[CandidateItem] = []
    for item in items:
        fatigue_hint = float(recent_counts.get(item.category_id, 0))
        if item.slot_type == "live_best_fatigued":
            fatigue_hint += 1.0

        rewritten.append(
            item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "fatigue_hint": float(round(fatigue_hint, 4)),
                    }
                }
            )
        )

    return rewritten