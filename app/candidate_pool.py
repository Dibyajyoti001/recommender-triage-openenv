from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import CandidateItem
from .tasks import CATEGORY_NAMES, K, get_task_config


def _normalize(v: Sequence[float]) -> List[float]:
    arr = np.asarray(v, dtype=float)
    arr = np.maximum(arr, 0.0)
    s = float(arr.sum())
    if s <= 0:
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / s
    return [float(x) for x in arr]


def _uniform(k: int) -> List[float]:
    return [1.0 / k for _ in range(k)]


def _style_vec(rng: np.random.Generator) -> List[float]:
    v = rng.uniform(0.0, 1.0, size=4)
    s = float(v.sum())
    return [float(x / s) for x in v]


def _title(category_name: str, freshness: str, slot_type: str, idx: int) -> str:
    return f"{category_name} :: {freshness} :: {slot_type} :: {idx}"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sample_quality(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sample_resource(
    rng: np.random.Generator,
    *,
    center: float,
    spread: float = 0.08,
) -> float:
    return float(_clip01(rng.normal(center, spread)))


def _primary_category(topic_vector: Sequence[float]) -> int:
    return int(np.argmax(np.asarray(topic_vector, dtype=float)))


def _least_recent_category(history_categories: Sequence[int]) -> int:
    if not history_categories:
        return 0
    last_seen = {c: -10_000 for c in range(K)}
    for idx, cat in enumerate(history_categories):
        last_seen[cat] = idx
    return min(last_seen, key=last_seen.get)


def _anti_distribution(v: Sequence[float]) -> List[float]:
    arr = np.asarray(v, dtype=float)
    arr = 1.0 - arr
    arr = np.maximum(arr, 1e-6)
    arr = arr / float(arr.sum())
    return [float(x) for x in arr]


def _to_sparse(v: Sequence[float], max_topics: int = 3, min_keep: float = 0.10) -> List[float]:
    arr = np.asarray(v, dtype=float)
    arr = np.maximum(arr, 0.0)

    if arr.sum() <= 0:
        arr = np.ones_like(arr)

    top_idx = np.argsort(arr)[::-1][:max_topics]
    mask = np.zeros_like(arr, dtype=bool)
    mask[top_idx] = True
    mask = np.logical_or(mask, arr >= min_keep)

    sparse = np.where(mask, arr, 0.0)
    if sparse.sum() <= 0:
        sparse[np.argmax(arr)] = arr[np.argmax(arr)]

    sparse = sparse / float(sparse.sum())
    return [float(x) for x in sparse]


def _blend_sparse(*parts: Sequence[float]) -> List[float]:
    arr = np.sum(np.asarray(parts, dtype=float), axis=0)
    return _to_sparse(arr, max_topics=3, min_keep=0.10)


def _make_item(
    *,
    item_id: int,
    topic_vector: Sequence[float],
    quality: float,
    cost: float,
    risk: float,
    latency: float,
    freshness: str,
    slot_type: str,
    rng: np.random.Generator,
    metadata: Dict[str, float | int | str | List[float]],
) -> CandidateItem:
    x = _to_sparse(topic_vector, max_topics=3, min_keep=0.10)
    category_id = _primary_category(x)
    category_name = CATEGORY_NAMES[category_id]
    engagement = _clip01(0.55 * quality + 0.45 * float(rng.uniform(0.35, 0.95)))
    return CandidateItem(
        item_id=item_id,
        title=_title(category_name, freshness, slot_type, item_id),
        category_id=category_id,
        category_name=category_name,
        quality=float(round(quality, 4)),
        engagement=float(round(engagement, 4)),
        cost=float(round(cost, 4)),
        risk=float(round(risk, 4)),
        latency=float(round(latency, 4)),
        freshness=freshness,
        style_vector=_style_vec(rng),
        slot_type=slot_type,
        metadata=metadata,
        topic_vector=x,
    )


def build_candidate_pool(
    *,
    task_id: str,
    turn: int,
    alpha: float,
    eta_q: float,
    z: Sequence[float],
    m: Sequence[float],
    item_fatigue: Optional[Dict[int, float]],
    history_categories: Sequence[int],
    rng: np.random.Generator,
    H_topic: Optional[Sequence[float]] = None,
    F_topic: Optional[Sequence[float]] = None,
    trust_level: float = 0.72,
    budget_remaining: float = 1.0,
    risk_tolerance: float = 1.0,
    latency_budget: float = 1.0,
    diversity_collapsed: bool = False,
) -> List[CandidateItem]:
    cfg = get_task_config(task_id)

    z_vec = _normalize(z)
    m_vec = _normalize(m)
    h_vec = _normalize(H_topic if H_topic is not None else _uniform(K))
    f_raw = F_topic if F_topic is not None and sum(F_topic) > 1e-8 else _uniform(K)
    f_vec = _normalize(f_raw)
    u_vec = _uniform(K)
    anti_m = _anti_distribution(m_vec)
    anti_z = _anti_distribution(z_vec)
    trust = _clip01(trust_level)
    trust_repair = max(0.0, 0.65 - trust)
    budget_pressure = max(0.0, 0.60 - _clip01(budget_remaining))
    latency_pressure = max(0.0, 0.60 - _clip01(latency_budget))
    safety_pressure = max(0.0, 0.60 - _clip01(risk_tolerance))

    k_z = int(np.argmax(np.asarray(z_vec)))
    k_m = int(np.argmax(np.asarray(m_vec)))
    least_recent = _least_recent_category(history_categories)
    least_recent_oh = [1.0 if i == least_recent else 0.0 for i in range(K)]

    balanced = _blend_sparse(
        [(1.0 - cfg.omega_mix - 0.10 * trust_repair) * z_vec[i] for i in range(K)],
        [cfg.omega_mix * m_vec[i] for i in range(K)],
        [0.10 * trust_repair * least_recent_oh[i] for i in range(K)],
    )

    fatigue_trap = _blend_sparse(
        [0.60 * z_vec[i] for i in range(K)],
        [0.25 * f_vec[i] for i in range(K)],
        [0.15 * h_vec[i] for i in range(K)],
    )

    exploration = _blend_sparse(
        [(0.45 + 0.10 * trust_repair) * z_vec[i] for i in range(K)],
        [(0.35 - 0.10 * trust_repair) * (1.0 - f_vec[i]) for i in range(K)],
        [0.20 * least_recent_oh[i] for i in range(K)],
    )

    if task_id == "task_3":
        s = cfg.conflict_strength
        conflict = _blend_sparse(
            [(1.0 - s) * u_vec[i] for i in range(K)],
            [s * anti_m[i] for i in range(K)],
        )
    else:
        s = min(0.55, cfg.conflict_strength)
        conflict = _blend_sparse(
            [(1.0 - s) * anti_z[i] for i in range(K)],
            [s * anti_m[i] for i in range(K)],
        )

    live_vec = _blend_sparse(
        [(0.70 - 0.10 * trust_repair) * z_vec[i] for i in range(K)],
        [0.10 * trust_repair * m_vec[i] for i in range(K)],
        [0.20 * least_recent_oh[i] for i in range(K)],
        [0.10 * u_vec[i] for i in range(K)],
    )

    mem_vec = _blend_sparse(
        [0.66 * m_vec[i] for i in range(K)],
        [0.20 * u_vec[i] for i in range(K)],
        [0.14 * z_vec[i] for i in range(K)],
    )

    if task_id == "task_1" and rng.uniform() < 0.70:
        live_vec = _blend_sparse(
            [0.78 * m_vec[i] for i in range(K)],
            [0.22 * z_vec[i] for i in range(K)],
        )

    if task_id == "task_2":
        fatigue_trap = _blend_sparse(
            [0.70 * z_vec[i] for i in range(K)],
            [0.20 * h_vec[i] for i in range(K)],
            [0.10 * f_vec[i] for i in range(K)],
        )

    if task_id == "task_4":
        fatigue_trap = _blend_sparse(
            [0.76 * z_vec[i] for i in range(K)],
            [0.18 * h_vec[i] for i in range(K)],
            [0.06 * f_vec[i] for i in range(K)],
        )
        exploration = _blend_sparse(
            [0.35 * z_vec[i] for i in range(K)],
            [0.35 * (1.0 - h_vec[i]) for i in range(K)],
            [0.30 * least_recent_oh[i] for i in range(K)],
        )

    if diversity_collapsed:
        live_vec = _blend_sparse(
            [0.45 * z_vec[i] for i in range(K)],
            [0.30 * (1.0 - h_vec[i]) for i in range(K)],
            [0.25 * least_recent_oh[i] for i in range(K)],
        )
        mem_vec = _blend_sparse(
            [0.46 * m_vec[i] for i in range(K)],
            [0.28 * least_recent_oh[i] for i in range(K)],
            [0.26 * u_vec[i] for i in range(K)],
        )

    next_id_base = turn * 10_000 + 100
    items: List[CandidateItem] = []

    item_specs = [
        ("live_best_fresh", live_vec, _sample_quality(rng, 0.75, 0.95), "fresh"),
        ("memory_best_fresh", mem_vec, _sample_quality(rng, 0.70, 0.90), "fresh"),
        ("balanced_bridge", balanced, _sample_quality(rng, 0.64, 0.84), "fresh"),
        ("fatigue_trap", fatigue_trap, _sample_quality(rng, 0.62, 0.82), "stale"),
        ("exploration_option", exploration, _sample_quality(rng, 0.45, 0.68), "novel"),
        ("conflict_option", conflict, _sample_quality(rng, 0.56, 0.80), "fresh"),
    ]

    for idx, (slot_type, topic_vec, quality, freshness) in enumerate(item_specs, start=1):
        topic_vec_sparse = _to_sparse(topic_vec, max_topics=3, min_keep=0.10)
        fatigue_hint = float(sum(float(a) * float(b) for a, b in zip(f_vec, topic_vec_sparse)))
        repetition_hint = float(sum(float(a) * float(b) for a, b in zip(h_vec, topic_vec_sparse)))
        resource_profiles = {
            "live_best_fresh": (0.70, 0.55, 0.58),
            "memory_best_fresh": (0.44, 0.22, 0.28),
            "balanced_bridge": (0.52, 0.30, 0.40),
            "fatigue_trap": (0.60, 0.68, 0.34),
            "exploration_option": (0.40, 0.62, 0.55),
            "conflict_option": (0.35, 0.52, 0.46),
        }
        base_cost, base_risk, base_latency = resource_profiles[slot_type]

        if freshness == "novel":
            base_risk += 0.06
            base_latency += 0.05
        elif freshness == "stale":
            base_risk += 0.04
            base_latency -= 0.04

        if slot_type == "live_best_fresh":
            base_cost += 0.08 * budget_pressure
            base_risk += 0.06 * safety_pressure
        if slot_type == "exploration_option":
            base_risk += 0.08 * safety_pressure
        if slot_type == "fatigue_trap":
            base_risk += 0.08 * trust_repair
        if diversity_collapsed and slot_type in {"live_best_fresh", "memory_best_fresh"}:
            base_risk += 0.06
            base_cost += 0.05

        cost = _sample_resource(rng, center=base_cost, spread=0.07)
        risk = _sample_resource(rng, center=base_risk, spread=0.07)
        latency = _sample_resource(rng, center=base_latency + 0.06 * latency_pressure, spread=0.06)

        items.append(
            _make_item(
                item_id=next_id_base + idx,
                topic_vector=topic_vec_sparse,
                quality=quality,
                cost=cost,
                risk=risk,
                latency=latency,
                freshness=freshness,
                slot_type=slot_type,
                rng=rng,
                metadata={
                    "target": slot_type,
                    "fatigue_hint": float(round(fatigue_hint, 4)),
                    "repetition_hint": float(round(repetition_hint, 4)),
                    "topic_vector": topic_vec_sparse,
                    "live_category": k_z,
                    "memory_category": k_m,
                    "topic_order": CATEGORY_NAMES,
                    "cost_hint": float(round(cost, 4)),
                    "risk_hint": float(round(risk, 4)),
                    "latency_hint": float(round(latency, 4)),
                },
            )
        )

    return items
