from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import List, Optional, Sequence, Tuple

from .models import CounterfactualAuditResult, FinalGradeBreakdown
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
    trust_before: float = 0.7
    trust_after: float = 0.7
    observed_feedback: float = 0.5
    feedback_volatility: float = 0.0
    repetition_pressure: float = 0.0
    novelty_violation: float = 0.0
    resource_pressure: float = 0.0
    risk_exposure: float = 0.0
    diversity_pressure: float = 0.0
    platform_gain: float = 0.0
    budget_remaining: float = 1.0
    risk_tolerance: float = 1.0
    latency_budget: float = 1.0
    calibration_target: float = 1.0
    slot_type: str = "live_best_fresh"
    saturated: bool = False
    trust_collapsed: bool = False
    risk_collapsed: bool = False
    diversity_collapsed: bool = False
    exploration_flag: bool = False
    confidence_score: float = 0.8


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
    if task_id == "task_4":
        return echo_chamber_adaptation_grade(traj)

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


def trust_grade(traj: Sequence[TrajectoryStep]) -> float:
    if not traj:
        return 0.0
    mean_trust = sum(step.trust_after for step in traj) / len(traj)
    terminal_trust = traj[-1].trust_after
    return _clip01(0.60 * mean_trust + 0.40 * terminal_trust)


def calibration_grade(traj: Sequence[TrajectoryStep]) -> float:
    if not traj:
        return 0.0
    scores = [
        exp(-abs(step.confidence_score - step.calibration_target))
        for step in traj
    ]
    return _clip01(sum(scores) / len(scores))


def risk_safety_grade(traj: Sequence[TrajectoryStep]) -> float:
    if not traj:
        return 0.0
    mean_risk = sum(step.risk_exposure for step in traj) / len(traj)
    return _clip01(1.0 - mean_risk)


def resource_efficiency_grade(traj: Sequence[TrajectoryStep]) -> float:
    if not traj:
        return 0.0
    mean_pressure = sum(step.resource_pressure for step in traj) / len(traj)
    return _clip01(1.0 - mean_pressure)


def collapse_gate(traj: Sequence[TrajectoryStep]) -> float:
    gate = 1.0
    if any(step.trust_collapsed for step in traj):
        gate *= 0.45
    if any(step.risk_collapsed for step in traj):
        gate *= 0.55
    if any(step.diversity_collapsed for step in traj):
        gate *= 0.60
    return _clip01(gate)


def floor_gate(trust_score: float, risk_safe_score: float, diversity_score: float) -> float:
    gate = min(
        1.0,
        trust_score / 0.35 if trust_score > 0.0 else 0.0,
        risk_safe_score / 0.35 if risk_safe_score > 0.0 else 0.0,
        diversity_score / 0.30 if diversity_score > 0.0 else 0.0,
    )
    return _clip01(max(0.20, gate))


def counterfactual_audit_grade(
    audits: Sequence[CounterfactualAuditResult] | None,
) -> Tuple[float, int]:
    if not audits:
        return 0.0, 0
    score = sum(audit.audit_score for audit in audits) / len(audits)
    return _clip01(score), len(audits)


def echo_chamber_adaptation_grade(
    traj: Sequence[TrajectoryStep],
) -> Tuple[float, Optional[int], Optional[int]]:
    if not traj:
        return 0.0, None, None

    onset = next((idx for idx, step in enumerate(traj) if step.saturated), None)
    if onset is None:
        if len(traj) <= 1:
            return _clip01(trust_grade(traj)), None, None

        switches = sum(
            1
            for i in range(1, len(traj))
            if traj[i].chosen_category_id != traj[i - 1].chosen_category_id
        )
        switch_rate = switches / (len(traj) - 1)
        exploratory_rate = sum(
            1
            for step in traj
            if step.exploration_flag or step.slot_type == "exploration_option"
        ) / len(traj)
        prevention_score = (
            0.45 * trust_grade(traj)
            + 0.30 * switch_rate
            + 0.25 * exploratory_rate
        )
        return _clip01(prevention_score), None, None

    recovery = None
    for idx in range(onset + 2, len(traj)):
        trust_avg = sum(traj[j].trust_after for j in range(idx - 1, idx + 1)) / 2.0
        nov_avg = sum(traj[j].novelty_violation for j in range(idx - 1, idx + 1)) / 2.0
        if trust_avg >= 0.62 and nov_avg <= 0.05:
            recovery = idx
            break

    pre_window = traj[max(0, onset - 4) : onset + 1]
    if len(pre_window) <= 1:
        switch_rate = 0.0
    else:
        switches = sum(
            1
            for i in range(1, len(pre_window))
            if pre_window[i].chosen_category_id != pre_window[i - 1].chosen_category_id
        )
        switch_rate = switches / (len(pre_window) - 1)

    exploratory_rate = sum(
        1
        for step in pre_window
        if step.exploration_flag or step.slot_type == "exploration_option"
    ) / max(1, len(pre_window))

    if recovery is None:
        stability = sum(step.trust_after for step in traj[onset:]) / max(1, len(traj) - onset)
        score = 0.45 * switch_rate + 0.25 * exploratory_rate + 0.30 * stability
        return _clip01(score), onset, None

    delay_score = exp(-0.40 * (recovery - onset))
    score = 0.25 * switch_rate + 0.20 * exploratory_rate + 0.55 * delay_score
    return _clip01(score), onset, recovery


def final_grade(
    traj: Sequence[TrajectoryStep],
    task_id: str,
    audits: Sequence[CounterfactualAuditResult] | None = None,
) -> FinalGradeBreakdown:
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
    t = trust_grade(traj)
    c = calibration_grade(traj)
    r = risk_safety_grade(traj)
    e = resource_efficiency_grade(traj)
    audit_score, audit_count = counterfactual_audit_grade(audits)

    if task_id == "task_4":
        base = 0.30 * s + 0.25 * t + 0.20 * r + 0.25 * d
        gate = collapse_gate(traj)
        floor = floor_gate(t, r, d)
        final = _clip01(base * gate * floor)
        if audit_count > 0 and cfg.counterfactual_audit_weight > 0.0:
            final = _clip01(
                (1.0 - cfg.counterfactual_audit_weight) * final
                + cfg.counterfactual_audit_weight * audit_score
            )
    elif task_id == "task_5":
        base = 0.25 * s + 0.20 * t + 0.20 * r + 0.10 * d + 0.25 * c
        gate = collapse_gate(traj)
        floor = floor_gate(t, r, max(d, 0.30 * c))
        final = _clip01(base * gate * floor)
        if audit_count > 0 and cfg.counterfactual_audit_weight > 0.0:
            final = _clip01(
                (1.0 - cfg.counterfactual_audit_weight) * final
                + cfg.counterfactual_audit_weight * audit_score
            )
    else:
        gate = 1.0
        floor = 1.0
        final = (
            cfg.omega_1 * s
            + cfg.omega_2 * d
            + cfg.omega_3 * a
            + cfg.omega_4 * m
            + cfg.omega_trust * t
            + cfg.omega_calibration * c
        )
        final = _clip01(final)

    return FinalGradeBreakdown(
        satisfaction=float(round(s, 6)),
        diversity=float(round(d, 6)),
        adaptation=float(round(a, 6)),
        memory_use=float(round(m, 6)),
        trust=float(round(t, 6)),
        calibration=float(round(c, 6)),
        risk_safety=float(round(r, 6)),
        resource_efficiency=float(round(e, 6)),
        counterfactual_audit=float(round(audit_score, 6)),
        audited_turns=audit_count,
        collapse_gate=float(round(gate, 6)),
        floor_gate=float(round(floor, 6)),
        final_score=float(round(final, 6)),
        drift_turn=drift_turn,
        recovered_turn=recovery_turn,
        task_weights={
            "satisfaction": cfg.omega_1,
            "diversity": cfg.omega_2,
            "adaptation": cfg.omega_3,
            "memory_use": cfg.omega_4,
            "trust": cfg.omega_trust,
            "calibration": cfg.omega_calibration,
        },
    )
