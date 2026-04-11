from __future__ import annotations

from app.graders import (
    TrajectoryStep,
    adaptation_grade,
    calibration_grade,
    counterfactual_audit_grade,
    detect_drift_turn,
    detect_recovery_turn,
    diversity_grade,
    echo_chamber_adaptation_grade,
    final_grade,
    memory_use_grade,
    risk_safety_grade,
    satisfaction_grade,
    trust_grade,
)
from app.models import CounterfactualAuditResult


def make_step(
    turn_id: int,
    chosen_category_id: int,
    relevance: float,
    satisfaction_proxy: float,
    memory_confidence: float,
    m: list[float],
    z: list[float],
    p_before: float,
    p_after: float,
    exploration_flag: bool = False,
    confidence_score: float = 0.8,
) -> TrajectoryStep:
    return TrajectoryStep(
        turn_id=turn_id,
        chosen_item_id=1000 + turn_id,
        chosen_category_id=chosen_category_id,
        chosen_category_name="X",
        relevance=relevance,
        satisfaction_proxy=satisfaction_proxy,
        memory_confidence=memory_confidence,
        m=m,
        z=z,
        p_before=p_before,
        p_after=p_after,
        exploration_flag=exploration_flag,
        confidence_score=confidence_score,
    )


def test_satisfaction_grade_basic_average():
    traj = [
        make_step(0, 0, 0.8, 0.6, 0.8, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1.0, 0.9),
        make_step(1, 0, 0.8, 0.8, 0.8, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.9, 0.8),
    ]
    assert abs(satisfaction_grade(traj) - 0.7) < 1e-6


def test_diversity_grade_in_range():
    traj = [
        make_step(0, 0, 0.7, 0.7, 0.8, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1.0, 0.9),
        make_step(1, 1, 0.7, 0.7, 0.8, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 0.9, 0.8),
        make_step(2, 2, 0.7, 0.7, 0.8, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 0.8, 0.7),
    ]
    score = diversity_grade(traj, nu=0.5)
    assert 0.0 <= score <= 1.0


def test_adaptation_no_drift_returns_one_for_non_conflict_task():
    base_m = [0.36, 0.22, 0.14, 0.06, 0.03, 0.08, 0.03, 0.02, 0.03, 0.03]
    base_z = [0.32, 0.24, 0.16, 0.06, 0.03, 0.07, 0.03, 0.02, 0.03, 0.04]

    traj = [
        make_step(0, 0, 0.72, 0.70, 0.85, base_m, base_z, 1.0, 0.95),
        make_step(1, 0, 0.71, 0.69, 0.84, base_m, base_z, 0.95, 0.90),
    ]

    score, drift, recovery = adaptation_grade(
        traj,
        task_id="task_1",
        tau=0.35,
        theta_rec=0.60,
        lambda_A=0.45,
    )

    assert score == 1.0
    assert drift is None
    assert recovery is None


def test_task3_detects_drift_and_no_recovery():
    m = [0.04, 0.06, 0.08, 0.10, 0.04, 0.26, 0.18, 0.04, 0.14, 0.06]
    z_pre = [0.28, 0.30, 0.18, 0.06, 0.02, 0.04, 0.02, 0.02, 0.04, 0.04]
    z_post = [0.02, 0.03, 0.04, 0.04, 0.02, 0.28, 0.24, 0.05, 0.18, 0.10]

    traj = [
        make_step(0, 0, 0.72, 0.71, 0.90, m, z_pre, 1.0, 0.95),
        make_step(1, 0, 0.68, 0.67, 0.88, m, z_pre, 0.95, 0.90),
        make_step(2, 0, 0.60, 0.58, 0.84, m, z_pre, 0.90, 0.82),
        make_step(3, 0, 0.50, 0.49, 0.78, m, z_pre, 0.82, 0.76),
        make_step(4, 1, 0.42, 0.44, 0.70, m, z_post, 0.76, 0.60),
        make_step(5, 1, 0.41, 0.43, 0.66, m, z_post, 0.60, 0.58),
        make_step(6, 1, 0.39, 0.40, 0.62, m, z_post, 0.58, 0.55),
    ]

    drift_turn = detect_drift_turn(traj, task_id="task_3", tau=0.35)
    recovery_turn = detect_recovery_turn(traj, drift_turn, theta_rec=0.60)
    score, drift, recovery = adaptation_grade(
        traj,
        task_id="task_3",
        tau=0.35,
        theta_rec=0.60,
        lambda_A=0.45,
    )

    assert drift_turn is not None
    assert recovery_turn is None
    assert drift == drift_turn
    assert recovery is None
    assert score == 0.0


def test_memory_use_grade_in_range():
    m = [0.36, 0.22, 0.14, 0.06, 0.03, 0.08, 0.03, 0.02, 0.03, 0.03]
    z = [0.32, 0.24, 0.16, 0.06, 0.03, 0.07, 0.03, 0.02, 0.03, 0.04]

    traj = [
        make_step(0, 0, 0.72, 0.70, 0.85, m, z, 1.0, 0.95),
        make_step(1, 1, 0.65, 0.66, 0.80, m, z, 0.95, 0.90),
        make_step(2, 0, 0.68, 0.67, 0.75, m, z, 0.90, 0.85),
    ]

    score = memory_use_grade(traj)
    assert 0.0 <= score <= 1.0


def test_trust_and_calibration_grades_in_range():
    m = [0.2] + [0.0888888889] * 9
    z = [0.2] + [0.0888888889] * 9
    traj = [
        make_step(0, 0, 0.70, 0.68, 0.80, m, z, 1.0, 0.92, confidence_score=0.88),
        make_step(1, 1, 0.66, 0.64, 0.76, m, z, 0.92, 0.88, confidence_score=0.72),
    ]
    traj[0].trust_after = 0.78
    traj[1].trust_after = 0.74
    traj[0].calibration_target = 0.90
    traj[1].calibration_target = 0.70

    assert 0.0 <= trust_grade(traj) <= 1.0
    assert 0.0 <= calibration_grade(traj) <= 1.0
    assert 0.0 <= risk_safety_grade(traj) <= 1.0


def test_task4_echo_chamber_adaptation_runs():
    m = [0.02, 0.03, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.55, 0.07]
    z = [0.02, 0.03, 0.04, 0.04, 0.03, 0.08, 0.04, 0.07, 0.58, 0.07]
    traj = [
        make_step(0, 8, 0.80, 0.78, 0.88, m, z, 1.0, 0.96, exploration_flag=False),
        make_step(1, 8, 0.79, 0.77, 0.87, m, z, 0.96, 0.92, exploration_flag=False),
        make_step(2, 5, 0.62, 0.61, 0.86, m, z, 0.92, 0.88, exploration_flag=True),
        make_step(3, 8, 0.54, 0.48, 0.84, m, z, 0.88, 0.70, exploration_flag=False),
    ]
    traj[0].trust_after = 0.82
    traj[1].trust_after = 0.75
    traj[2].trust_after = 0.72
    traj[3].trust_after = 0.58
    traj[3].saturated = True
    traj[3].novelty_violation = 0.12
    score, onset, recovery = echo_chamber_adaptation_grade(traj)

    assert 0.0 <= score <= 1.0
    assert onset is not None
    assert recovery is None or recovery >= onset


def test_final_grade_task3_runs_and_returns_valid_fields():
    m = [0.04, 0.06, 0.08, 0.10, 0.04, 0.26, 0.18, 0.04, 0.14, 0.06]
    z_pre = [0.28, 0.30, 0.18, 0.06, 0.02, 0.04, 0.02, 0.02, 0.04, 0.04]
    z_post = [0.02, 0.03, 0.04, 0.04, 0.02, 0.28, 0.24, 0.05, 0.18, 0.10]

    traj = [
        make_step(0, 0, 0.72, 0.71, 0.90, m, z_pre, 1.0, 0.95),
        make_step(1, 0, 0.68, 0.67, 0.88, m, z_pre, 0.95, 0.90),
        make_step(2, 0, 0.60, 0.58, 0.84, m, z_pre, 0.90, 0.82),
        make_step(3, 0, 0.50, 0.49, 0.78, m, z_pre, 0.82, 0.76),
        make_step(4, 1, 0.42, 0.44, 0.70, m, z_post, 0.76, 0.60),
        make_step(5, 1, 0.41, 0.43, 0.66, m, z_post, 0.60, 0.58),
        make_step(6, 1, 0.39, 0.40, 0.62, m, z_post, 0.58, 0.55),
    ]

    out = final_grade(traj, "task_3")

    assert 0.0 <= out.satisfaction <= 1.0
    assert 0.0 <= out.diversity <= 1.0
    assert 0.0 <= out.adaptation <= 1.0
    assert 0.0 <= out.memory_use <= 1.0
    assert 0.0 <= out.trust <= 1.0
    assert 0.0 <= out.calibration <= 1.0
    assert 0.0 <= out.risk_safety <= 1.0
    assert 0.0 <= out.resource_efficiency <= 1.0
    assert 0.0 <= out.final_score <= 1.0


def test_task5_non_compensatory_gate_penalizes_collapse():
    m = [0.12, 0.16, 0.09, 0.08, 0.07, 0.10, 0.09, 0.11, 0.10, 0.08]
    z = [0.10, 0.18, 0.10, 0.08, 0.08, 0.08, 0.08, 0.12, 0.10, 0.08]
    traj = [
        make_step(0, 1, 0.78, 0.74, 0.76, m, z, 1.0, 0.92, confidence_score=0.90),
        make_step(1, 1, 0.76, 0.72, 0.74, m, z, 0.92, 0.88, confidence_score=0.92),
    ]
    traj[0].trust_after = 0.18
    traj[1].trust_after = 0.16
    traj[0].risk_exposure = 0.82
    traj[1].risk_exposure = 0.86
    traj[0].diversity_pressure = 0.84
    traj[1].diversity_pressure = 0.88
    traj[0].resource_pressure = 0.72
    traj[1].resource_pressure = 0.78
    traj[0].trust_collapsed = True
    traj[0].risk_collapsed = True
    traj[0].diversity_collapsed = True
    traj[0].calibration_target = 0.40
    traj[1].calibration_target = 0.35

    out = final_grade(traj, "task_5")

    assert out.collapse_gate < 1.0
    assert out.floor_gate <= 1.0
    assert out.final_score < 0.30


def test_counterfactual_audit_grade_in_range():
    audits = [
        CounterfactualAuditResult(
            turn_id=7,
            fragility=0.62,
            chosen_value=0.71,
            safe_value=0.58,
            risky_value=0.49,
            safe_advantage=0.13,
            risky_advantage=0.22,
            audit_score=0.81,
            chosen_slot_type="balanced_bridge",
            safe_slot_type="memory_best_fresh",
            risky_slot_type="live_best_fresh",
        ),
        CounterfactualAuditResult(
            turn_id=12,
            fragility=0.74,
            chosen_value=0.64,
            safe_value=0.60,
            risky_value=0.52,
            safe_advantage=0.04,
            risky_advantage=0.12,
            audit_score=0.66,
            chosen_slot_type="exploration_option",
            safe_slot_type="balanced_bridge",
            risky_slot_type="fatigue_trap",
        ),
    ]

    score, count = counterfactual_audit_grade(audits)

    assert count == 2
    assert 0.0 <= score <= 1.0


def test_task5_final_grade_blends_counterfactual_audit():
    m = [0.12, 0.16, 0.09, 0.08, 0.07, 0.10, 0.09, 0.11, 0.10, 0.08]
    z = [0.10, 0.18, 0.10, 0.08, 0.08, 0.08, 0.08, 0.12, 0.10, 0.08]
    traj = [
        make_step(0, 1, 0.78, 0.74, 0.76, m, z, 1.0, 0.92, confidence_score=0.90),
        make_step(1, 1, 0.76, 0.72, 0.74, m, z, 0.92, 0.88, confidence_score=0.92),
    ]
    traj[0].trust_after = 0.70
    traj[1].trust_after = 0.72
    traj[0].risk_exposure = 0.18
    traj[1].risk_exposure = 0.20
    traj[0].calibration_target = 0.84
    traj[1].calibration_target = 0.80

    no_audit = final_grade(traj, "task_5")
    with_audit = final_grade(
        traj,
        "task_5",
        [
            CounterfactualAuditResult(
                turn_id=8,
                fragility=0.68,
                chosen_value=0.78,
                safe_value=0.60,
                risky_value=0.56,
                safe_advantage=0.18,
                risky_advantage=0.22,
                audit_score=0.86,
                chosen_slot_type="balanced_bridge",
                safe_slot_type="memory_best_fresh",
                risky_slot_type="live_best_fresh",
            )
        ],
    )

    assert with_audit.audited_turns == 1
    assert with_audit.counterfactual_audit > 0.0
    assert with_audit.final_score > no_audit.final_score
