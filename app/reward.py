from __future__ import annotations

from math import exp
from typing import Dict, List, Sequence, Tuple

from .models import CandidateItem, RewardBreakdown
from .tasks import GLOBAL


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clip_reward(x: float) -> float:
    return max(-1.0, min(1.0, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _one_hot(category_id: int, k: int) -> List[float]:
    v = [0.0] * k
    if 0 <= category_id < k:
        v[category_id] = 1.0
    return v


def _topic_vector(item: CandidateItem, k: int) -> List[float]:
    topic = getattr(item, "topic_vector", None) or item.metadata.get("topic_vector")
    if isinstance(topic, list) and len(topic) == k:
        cleaned = [max(0.0, float(x)) for x in topic]
        s = sum(cleaned)
        if s > 0:
            return [x / s for x in cleaned]
    return _one_hot(item.category_id, k)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def relevance(alpha: float, eta_q: float, z: Sequence[float], m: Sequence[float], item: CandidateItem) -> float:
    x = _topic_vector(item, len(z))
    return float((alpha * _dot(z, x) + (1.0 - alpha) * _dot(m, x) + eta_q * item.quality) / (1.0 + eta_q))


def category_fatigue_update(current: float, lambda_c: float, delta_c: float, chosen: bool) -> float:
    return lambda_c * current + (delta_c if chosen else 0.0)


def item_fatigue_update(current: float, lambda_i: float, delta_i: float, chosen: bool) -> float:
    return lambda_i * current + (delta_i if chosen else 0.0)


def fatigue_cost(
    category_fatigue_value: float,
    item_fatigue_value: float,
    *,
    w_c: float,
    w_i: float,
) -> float:
    return float(w_c * category_fatigue_value + w_i * item_fatigue_value)


def repetition_pressure(history_categories: Sequence[int], chosen_category: int, window: int) -> float:
    if window <= 0:
        return 0.0
    recent = list(history_categories[-(window - 1):]) + [chosen_category]
    same = sum(1 for c in recent if c == chosen_category)
    return float(same / float(window))


def novelty_violation(rho_t: float, nu: float) -> float:
    return float(max(0.0, rho_t - nu))


def update_patience(p_t: float, y_t: float) -> float:
    delta = GLOBAL.beta_patience * (2.0 * clip01(y_t) - 1.0)
    return clip01(p_t + delta)


def calibration_target(feedback_volatility: float, decay_k: float) -> float:
    return clip01(exp(-decay_k * max(0.0, feedback_volatility)))


def resource_pressure(
    item: CandidateItem,
    *,
    budget_remaining: float,
    risk_tolerance: float,
    latency_budget: float,
    eps: float = 1e-6,
) -> float:
    return clip01(
        0.45 * item.cost / max(budget_remaining, eps)
        + 0.30 * item.latency / max(latency_budget, eps)
        + 0.25 * item.risk / max(risk_tolerance, eps)
    )


def diversity_pressure(
    repetition_pressure_value: float,
    novelty_value: float,
    concentration_recent: float,
) -> float:
    return clip01(
        0.60 * repetition_pressure_value
        + 0.25 * novelty_value
        + 0.15 * concentration_recent
    )


def risk_exposure(
    confidence_overclaim: float,
    feedback_volatility: float,
    repetition_pressure_value: float,
    item_risk: float,
) -> float:
    return clip01(
        0.35 * confidence_overclaim
        + 0.25 * feedback_volatility
        + 0.20 * repetition_pressure_value
        + 0.20 * item_risk
    )


def satisfaction_proxy(
    relevance_value: float,
    fatigue_value: float,
    novelty_value: float,
    *,
    trust_level: float,
    trust_sensitivity: float,
    satisfaction_cap: float | None = None,
) -> float:
    trust_bonus = GLOBAL.zeta4 * trust_sensitivity * (clip01(trust_level) - 0.5)
    y = clip01(
        sigmoid(
            GLOBAL.zeta1 * relevance_value
            - GLOBAL.zeta2 * fatigue_value
            - GLOBAL.zeta3 * novelty_value
            + trust_bonus
        )
    )
    if satisfaction_cap is not None:
        y = min(y, satisfaction_cap)
    return y


def unnecessary_exploration_penalty(exploration_flag: bool, chi_t: float) -> float:
    if not exploration_flag:
        return 0.0
    return float(max(0.0, chi_t - GLOBAL.chi_threshold_explore_penalty))


def confidence_overclaim_penalty(confidence_score: float, y_t: float) -> float:
    return float(max(0.0, confidence_score - y_t))


def feedback_bucket(y_t: float) -> str:
    if y_t < 0.30:
        return "low"
    if y_t < 0.55:
        return "neutral"
    if y_t < 0.80:
        return "positive"
    return "strong_positive"


def pressure_bucket(rho_t: float) -> str:
    if rho_t < 0.34:
        return "low"
    if rho_t < 0.67:
        return "medium"
    return "high"


def confidence_bucket(chi_t: float) -> str:
    if chi_t < 0.34:
        return "weak"
    if chi_t < 0.67:
        return "moderate"
    return "strong"


def engagement_bucket(p_t: float) -> str:
    if p_t < 0.34:
        return "fragile"
    if p_t < 0.67:
        return "stable"
    return "engaged"


def _is_exploratory_item(item: CandidateItem) -> bool:
    return item.slot_type in {"exploration_option", "conflict_option"}


def compute_step_reward(
    *,
    alpha: float,
    eta_q: float,
    z: Sequence[float],
    m: Sequence[float],
    item: CandidateItem,
    category_fatigue_value: float,
    item_fatigue_value: float,
    history_categories: Sequence[int],
    nu: float,
    chi_t: float,
    exploration_flag: bool,
    confidence_score: float,
    repetition_window: int,
    w_c: float,
    w_i: float,
    H_topic: Sequence[float] | None = None,
    F_topic: Sequence[float] | None = None,
    trust_level: float = 0.5,
    trust_sensitivity: float = 1.0,
    feedback_volatility: float = 0.0,
    calibration_decay_k: float = 8.0,
    confidence_penalty_weight: float | None = None,
    budget_remaining: float = 1.0,
    risk_tolerance: float = 1.0,
    latency_budget: float = 1.0,
    concentration_recent: float = 0.0,
    satisfaction_cap: float | None = None,
) -> Tuple[float, RewardBreakdown, Dict[str, float]]:
    x = _topic_vector(item, len(z))
    r_t = relevance(alpha, eta_q, z, m, item)

    if F_topic is not None and len(F_topic) == len(x):
        c_fat = _dot(F_topic, x)
    else:
        c_fat = fatigue_cost(category_fatigue_value, item_fatigue_value, w_c=w_c, w_i=w_i)

    if H_topic is not None and len(H_topic) == len(x):
        rho_t = _dot(H_topic, x)
    else:
        rho_t = repetition_pressure(history_categories, item.category_id, repetition_window)

    v_nov = novelty_violation(rho_t, nu)
    y_t = satisfaction_proxy(
        r_t,
        c_fat,
        v_nov,
        trust_level=trust_level,
        trust_sensitivity=trust_sensitivity,
        satisfaction_cap=satisfaction_cap,
    )
    actual_exploration = bool(exploration_flag or _is_exploratory_item(item))
    u_t = unnecessary_exploration_penalty(actual_exploration, chi_t)
    p_conf = confidence_overclaim_penalty(confidence_score, y_t)
    alignment = _dot(z, m)
    c_target = calibration_target(feedback_volatility, calibration_decay_k)
    c_gap = abs(confidence_score - c_target)
    trust_bonus = GLOBAL.zeta4 * trust_sensitivity * (clip01(trust_level) - 0.5)
    conf_weight = GLOBAL.w_conf if confidence_penalty_weight is None else confidence_penalty_weight
    res_pressure = resource_pressure(
        item,
        budget_remaining=budget_remaining,
        risk_tolerance=risk_tolerance,
        latency_budget=latency_budget,
    )
    risk_value = risk_exposure(p_conf, feedback_volatility, rho_t, item.risk)
    div_pressure = diversity_pressure(rho_t, v_nov, concentration_recent)

    raw = (
        GLOBAL.w_r * r_t
        - GLOBAL.w_f * c_fat
        - GLOBAL.w_n * v_nov
        - GLOBAL.w_u * u_t
        - conf_weight * p_conf
    )
    clipped = clip_reward(raw)

    breakdown = RewardBreakdown(
        relevance=float(round(r_t, 6)),
        fatigue_cost=float(round(c_fat, 6)),
        novelty_violation=float(round(v_nov, 6)),
        unnecessary_exploration=float(round(u_t, 6)),
        confidence_penalty=float(round(p_conf, 6)),
        trust_bonus=float(round(trust_bonus, 6)),
        calibration_target=float(round(c_target, 6)),
        calibration_gap=float(round(c_gap, 6)),
        resource_pressure=float(round(res_pressure, 6)),
        risk_exposure=float(round(risk_value, 6)),
        diversity_pressure=float(round(div_pressure, 6)),
        raw_reward=float(round(raw, 6)),
        clipped_reward=float(round(clipped, 6)),
        satisfaction_proxy=float(round(y_t, 6)),
        repetition_pressure=float(round(rho_t, 6)),
        alignment=float(round(alignment, 6)),
    )
    aux = {
        "relevance": r_t,
        "fatigue_cost": c_fat,
        "repetition_pressure": rho_t,
        "novelty_violation": v_nov,
        "satisfaction_proxy": y_t,
        "alignment": alignment,
        "calibration_target": c_target,
        "calibration_gap": c_gap,
        "trust_bonus": trust_bonus,
        "resource_pressure": res_pressure,
        "risk_exposure": risk_value,
        "diversity_pressure": div_pressure,
    }
    return clipped, breakdown, aux
