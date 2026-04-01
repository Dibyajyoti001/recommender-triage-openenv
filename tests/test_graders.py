from app.graders import TrajectoryStep, adaptation_grade, memory_use_grade, final_grade
from app.tasks import get_task_config


def make_step(turn, cat, rel, sat, chi, m, z, p_before, p_after):
    return TrajectoryStep(
        turn_id=turn,
        chosen_item_id=turn + 100,
        chosen_category_id=cat,
        chosen_category_name="Documentary",
        relevance=rel,
        satisfaction_proxy=sat,
        memory_confidence=chi,
        m=m,
        z=z,
        p_before=p_before,
        p_after=p_after,
        exploration_flag=False,
        confidence_score=0.8,
    )


def test_memory_use_exact_rule():
    traj = [
        make_step(0, 0, 0.8, 0.8, 0.9, [0.7, 0.1, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05], 0.9, 0.8),
        make_step(1, 1, 0.5, 0.5, 0.4, [0.7, 0.1, 0.1, 0.05, 0.05], [0.2, 0.6, 0.1, 0.05, 0.05], 0.8, 0.7),
    ]
    score = memory_use_grade(traj)
    assert 0.0 <= score <= 1.0


def test_adaptation_no_drift_returns_one():
    traj = [
        make_step(0, 0, 0.7, 0.7, 0.8, [0.6, 0.2, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05], 0.9, 0.8),
        make_step(1, 0, 0.7, 0.7, 0.8, [0.6, 0.2, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05], 0.8, 0.75),
    ]
    score, drift, recovery = adaptation_grade(traj, tau=0.35, theta_rec=0.60, lambda_A=0.45)
    assert score == 1.0
    assert drift is None
    assert recovery is None


def test_final_grade_bounded():
    traj = [
        make_step(0, 0, 0.7, 0.7, 0.8, [0.6, 0.2, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05], 0.9, 0.8),
        make_step(1, 1, 0.6, 0.6, 0.7, [0.6, 0.2, 0.1, 0.05, 0.05], [0.2, 0.6, 0.1, 0.05, 0.05], 0.8, 0.3),
        make_step(2, 1, 0.7, 0.7, 0.6, [0.6, 0.2, 0.1, 0.05, 0.05], [0.2, 0.6, 0.1, 0.05, 0.05], 0.3, 0.25),
    ]
    breakdown = final_grade(traj, "task_3")
    assert 0.0 <= breakdown.final_score <= 1.0