from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Dict, List, Optional, Sequence

from .models import FinalGradeBreakdown
from .tasks import K, GLOBAL, TaskConfig, get_task_config


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class TrajectoryStep:
    turn_id: int
    chosen_item_id: int
    chosen_category_id: int
    chosen_category_name: str
    relevance: float
    satisfaction_proxy: float
    memory_confidence: float
    m: List[float]
    z: List[float]
    p_before: float
    p_after: float
    exploration_flag: bool
    confidence_score: float


def satisfaction_grade(traj: Sequence[TrajectoryStep]) -> float:
    if not traj:
        return 0.0
    return _clip01(sum(step.satisfaction_proxy for step in traj) / len(traj))


def diversity_grade(traj: Sequence[TrajectoryStep], nu: float) -> float:
    if not traj:
        return 0.0

    counts = [0] * K
    for step in traj:
        counts[step.chosen_category_id] += 1

    probs = [c / len(traj) for c in counts if c > 0]
    entropy = -sum(p * log(p) for p in probs)
    h_max = log(K)
    h_star = nu * h_max
    return _clip01(1.0 - min(1.0, abs(entropy - h_star) / h_max))


def detect_drift_turn(traj: Sequence[TrajectoryStep], tau: float) -> Optional[int]:
    """
    Exact frozen-spec drift rule:

    t* = first t where:
      p_t <= tau AND argmax(z_{t+1}) != argmax(z_t)

    Since each trajectory step stores z before the transition and p_before/p_after,
    we operationalize this as:
      p_after <= tau AND argmax(z_t_before) != argmax(z_{t+1}_before)

    where z_{t+1}_before is the next step's stored z.
    """
    if len(traj) < 2:
        return None

    for i in range(len(traj) - 1):
        curr = traj[i]
        nxt = traj[i + 1]
        curr_argmax = int(max(range(len(curr.z)), key=lambda j: curr.z[j]))
        next_argmax = int(max(range(len(nxt.z)), key=lambda j: nxt.z[j]))
        if curr.p_after <= tau and curr_argmax != next_argmax:
            return i
    return None


def detect_recovery_turn(
    traj: Sequence[TrajectoryStep],
    drift_turn: Optional[int],
    theta_rec: float,
) -> Optional[int]:
    """
    Exact frozen-spec recovery rule:

    recovery occurs at first t >= t* + 2 such that:
      avg_3(relevance) >= theta_rec
      avg_3(satisfaction_proxy) >= 0.60
    """
    if drift_turn is None:
        return None

    start = max(drift_turn + 2, 2)
    for idx in range(start, len(traj)):
        rel_avg = sum(traj[j].relevance for j in range(idx - 2, idx + 1)) / 3.0
        sat_avg = sum(traj[j].satisfaction_proxy for j in range(idx - 2, idx + 1)) / 3.0
        if rel_avg >= theta_rec and sat_avg >= 0.60:
            return idx
    return None


def adaptation_grade(
    traj: Sequence[TrajectoryStep],
    *,
    tau: float,
    theta_rec: float,
    lambda_A: float,
) -> tuple[float, Optional[int], Optional[int]]:
    drift_turn = detect_drift_turn(traj, tau=tau)
    if drift_turn is None:
        return 1.0, None, None

    recovery_turn = detect_recovery_turn(traj, drift_turn, theta_rec)
    if recovery_turn is None:
        return 0.0, drift_turn, None

    score = exp(-lambda_A * (recovery_turn - drift_turn))
    return _clip01(score), drift_turn, recovery_turn


def memory_use_grade(traj: Sequence[TrajectoryStep]) -> float:
    """
    Exact frozen-spec memory-use grader.

    Let:
      k_m = argmax_k m[k]
      k_z = argmax_k z_t[k]

    I_t = 1 if chi_t >= 0.60 and m[k_m] - z_t[k_z] >= -0.05 else 0
    J_t = 1 if chosen category == k_m else 0

    M(tau) = average_t 1[I_t == J_t]
    """
    if not traj:
        return 0.0

    matches = 0
    for step in traj:
        k_m = int(max(range(len(step.m)), key=lambda i: step.m[i]))
        k_z = int(max(range(len(step.z)), key=lambda i: step.z[i]))
        I_t = 1 if (step.memory_confidence >= 0.60 and (step.m[k_m] - step.z[k_z] >= -0.05)) else 0
        J_t = 1 if step.chosen_category_id == k_m else 0
        matches += 1 if I_t == J_t else 0

    return _clip01(matches / len(traj))


def final_grade(traj: Sequence[TrajectoryStep], task_id: str) -> FinalGradeBreakdown:
    cfg: TaskConfig = get_task_config(task_id)

    s = satisfaction_grade(traj)
    d = diversity_grade(traj, cfg.nu)
    a, drift_turn, recovery_turn = adaptation_grade(
        traj,
        tau=cfg.tau,
        theta_rec=GLOBAL.theta_rec,
        lambda_A=GLOBAL.lambda_A,
    )
    m = memory_use_grade(traj)

    final = (
        cfg.omega_1 * s
        + cfg.omega_2 * d
        + cfg.omega_3 * a
        + cfg.omega_4 * m
    )
    final = _clip01(final)

    return FinalGradeBreakdown(
        satisfaction=float(round(s, 6)),
        diversity=float(round(d, 6)),
        adaptation=float(round(a, 6)),
        memory_use=float(round(m, 6)),
        final_score=float(round(final, 6)),
        drift_turn=drift_turn,
        recovered_turn=recovery_turn,
        task_weights={
            "satisfaction": cfg.omega_1,
            "diversity": cfg.omega_2,
            "adaptation": cfg.omega_3,
            "memory_use": cfg.omega_4,
        },
    )