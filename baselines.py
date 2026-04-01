from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from app.models import Action, Observation
from app.simulator import RecommenderEnv


def history_only_action(obs: Observation) -> Action:
    top_memory_cat = obs.memory_summary.top_categories[0] if obs.memory_summary.top_categories else 0
    candidates = obs.candidate_items
    best = max(
        candidates,
        key=lambda x: (
            1 if x.category_id == top_memory_cat else 0,
            x.quality,
            x.engagement,
        ),
    )
    return Action(
        recommended_item_id=best.item_id,
        exploration_flag=False,
        confidence_score=0.80,
    )


def recent_only_action(obs: Observation) -> Action:
    if obs.recent_interactions:
        cats = [x.category_id for x in obs.recent_interactions[-5:]]
        majority_cat = Counter(cats).most_common(1)[0][0]
    else:
        majority_cat = obs.memory_summary.top_categories[0] if obs.memory_summary.top_categories else 0

    best = max(
        obs.candidate_items,
        key=lambda x: (
            1 if x.category_id == majority_cat else 0,
            x.quality,
            x.engagement,
        ),
    )
    return Action(
        recommended_item_id=best.item_id,
        exploration_flag=False,
        confidence_score=0.75,
    )


def greedy_action(obs: Observation) -> Action:
    rep_counts = obs.repetition_counts
    best = None
    best_score = float("-inf")
    for item in obs.candidate_items:
        rep_pen = 0.0
        if item.category_id < len(rep_counts):
            rep_pen = 0.10 * rep_counts[item.category_id]
        score = item.quality - 0.5 * rep_pen - 0.05 * (1 if item.freshness == "stale" else 0)
        if score > best_score:
            best_score = score
            best = item

    assert best is not None
    return Action(
        recommended_item_id=best.item_id,
        exploration_flag=False,
        confidence_score=0.70,
    )


def diversity_first_action(obs: Observation) -> Action:
    rep_counts = obs.repetition_counts
    best = None
    best_score = float("-inf")
    for item in obs.candidate_items:
        rep_pen = rep_counts[item.category_id] if item.category_id < len(rep_counts) else 0
        score = item.quality - 0.8 * rep_pen + (0.10 if item.freshness == "novel" else 0.0)
        if score > best_score:
            best_score = score
            best = item

    assert best is not None
    recent = [x.category_id for x in obs.recent_interactions[-2:]]
    majority = Counter(recent).most_common(1)[0][0] if recent else None
    explore = majority is not None and best.category_id != majority
    return Action(
        recommended_item_id=best.item_id,
        exploration_flag=bool(explore),
        confidence_score=0.60,
    )


def balanced_action(obs: Observation) -> Action:
    top_memory_cat = obs.memory_summary.top_categories[0] if obs.memory_summary.top_categories else 0
    recent_majority = None
    if obs.recent_interactions:
        cats = [x.category_id for x in obs.recent_interactions[-3:]]
        recent_majority = Counter(cats).most_common(1)[0][0]

    rep_counts = obs.repetition_counts
    best = None
    best_score = float("-inf")

    for item in obs.candidate_items:
        rep_pen = 0.08 * rep_counts[item.category_id] if item.category_id < len(rep_counts) else 0.0
        mem_bonus = 0.10 if item.category_id == top_memory_cat else 0.0
        live_bonus = 0.08 if (recent_majority is not None and item.category_id == recent_majority) else 0.0
        freshness_bonus = 0.05 if item.freshness == "fresh" else (0.04 if item.freshness == "novel" else -0.02)

        score = (
            0.45 * item.quality
            + 0.20 * item.engagement
            + mem_bonus
            + live_bonus
            + freshness_bonus
            - 0.20 * rep_pen
        )
        if score > best_score:
            best_score = score
            best = item

    assert best is not None
    explore = bool(obs.memory_confidence < 0.55 or obs.repetition_pressure_bucket == "high")
    return Action(
        recommended_item_id=best.item_id,
        exploration_flag=explore,
        confidence_score=0.80 if not explore else 0.65,
    )


BASELINES = {
    "history_only": history_only_action,
    "recent_only": recent_only_action,
    "greedy": greedy_action,
    "diversity_first": diversity_first_action,
    "balanced": balanced_action,
}


def evaluate_baseline(
    baseline_name: str,
    task_id: str,
    *,
    episodes: int = 10,
) -> Dict[str, float]:
    if baseline_name not in BASELINES:
        raise KeyError(f"Unknown baseline: {baseline_name}")

    policy = BASELINES[baseline_name]
    scores: List[float] = []

    for ep in range(episodes):
        env = RecommenderEnv(seed=1000 + ep)
        obs = env.reset(task_id=task_id, seed=2000 + ep)
        done = False

        while not done:
            action = policy(obs)
            step = env.step(action)
            obs = step.observation
            done = step.done

        scores.append(env.current_grade().final_score)

    return {
        "mean_score": round(sum(scores) / len(scores), 6),
        "min_score": round(min(scores), 6),
        "max_score": round(max(scores), 6),
    }