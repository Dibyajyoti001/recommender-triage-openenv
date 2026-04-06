from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import List, Optional, Sequence, Tuple

from .models import FinalGradeBreakdown
from .tasks import K, GLOBAL, TaskConfig, get_task_config


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _argmax(v: Sequence[float]) -> int:
    return int(max(range(len(v)), key=lambda i: v[i]))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _l1(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(abs(float(x) - float(y)) for x, y in zip(a, b)))


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
        if 0 <= step.chosen_category_id < K:
            counts[step.chosen_category_id] += 1

    probs = [c / len(traj) for c in counts if c > 0]
    if not probs:
        return 0.0

    entropy = -sum(p * log(p) for p in probs)
    h_max = log(K)
    h_star = nu * h_max
    return _clip01(1.0 - min(1.0, abs(entropy - h_star) / h_max))


def detect_drift_turn(
    traj: Sequence[TrajectoryStep],
    *,
    task_id: str,
    tau: float,
) -> Optional[int]:
    """
    Drift detection logic:

    Task 1 / Task 2:
        Keep the original conservative rule:
            p_after <= tau AND argmax(z_t) != argmax(z_{t+1})

    Task 3:
        Detect the first *real* live-intent shift between consecutive stored z states.
        Since the environment now injects a deliberate conflict event, we detect drift if:
            - argmax changes, OR
            - L1 shift in z is large enough, OR
            - alignment with memory drops enough
    """
    if len(traj) < 2:
        return None

    for i in range(len(traj) - 1):
        curr = traj[i]
        nxt = traj[i + 1]

        curr_argmax = _argmax(curr.z)
        next_argmax = _argmax(nxt.z)

        if task_id != "task_3":
            if curr.p_after <= tau and curr_argmax != next_argmax:
                return i
            continue

        # Task 3: robust drift detection
        z_shift = _l1(curr.z, nxt.z)
        mem_align_curr = _dot(curr.m, curr.z)
        mem_align_next = _dot(nxt.m, nxt.z)
        align_drop = mem_align_curr - mem_align_next

        # A real shift should not be tiny noise.
        if (
            curr_argmax != next_argmax
            or z_shift >= 0.30
            or align_drop >= 0.08
        ):
            return i

    return None


def detect_recovery_turn(
    traj: Sequence[TrajectoryStep],
    drift_turn: Optional[int],
    theta_rec: float,
) -> Optional[int]:
    """
    Recovery occurs at first t >= drift_turn + 2 such that:
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
    task_id: str,
    tau: float,
    theta_rec: float,
    lambda_A: float,
) -> Tuple[float, Optional[int], Optional[int]]:
    drift_turn = detect_drift_turn(traj, task_id=task_id, tau=tau)

    # For non-drift tasks, no drift is fine.
    if task_id != "task_3":
        if drift_turn is None:
            return 1.0, None, None

    # For Task 3, no drift means the environment/policy failed to surface the conflict.
    if task_id == "task_3" and drift_turn is None:
        return 0.0, None, None

    recovery_turn = detect_recovery_turn(traj, drift_turn, theta_rec)
    if recovery_turn is None:
        return 0.0, drift_turn, None

    score = exp(-lambda_A * (recovery_turn - drift_turn))
    return _clip01(score), drift_turn, recovery_turn


def memory_use_grade(traj: Sequence[TrajectoryStep]) -> float:
    """
    Memory-use grading:
      - reward using memory when memory is still trustworthy and aligned
      - reward NOT using memory when live intent has diverged

    We use:
      alignment_t = m · z_t

      I_t = 1 if memory should be trusted
          = 1 if (chi_t >= 0.60 and alignment_t >= 0.16)
            else 0

      J_t = 1 if chosen category == argmax(m)
            else 0

      Score = average_t 1[I_t == J_t]
    """
    if not traj:
        return 0.0

    matches = 0
    for step in traj:
        k_m = _argmax(step.m)
        alignment_t = _dot(step.m, step.z)

        I_t = 1 if (step.memory_confidence >= 0.60 and alignment_t >= 0.16) else 0
        J_t = 1 if step.chosen_category_id == k_m else 0

        if I_t == J_t:
            matches += 1

    return _clip01(matches / len(traj))


def final_grade(traj: Sequence[TrajectoryStep], task_id: str) -> FinalGradeBreakdown:
    cfg: TaskConfig = get_task_config(task_id)

    s = satisfaction_grade(traj)
    d = diversity_grade(traj, cfg.nu)
    a, drift_turn, recovery_turn = adaptation_grade(
        traj,
        task_id=task_id,
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