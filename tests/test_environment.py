from app.simulator import RecommendationPolicyEnvironment
from app.models import Action


def test_environment_reset_and_step():
    env = RecommendationPolicyEnvironment(seed=123)
    obs = env.reset("task_1", seed=456)
    assert obs.task_id == "task_1"
    assert len(obs.candidate_items) == 6
    assert 0.0 <= obs.trust_signal <= 1.0
    assert 0.0 <= obs.engagement_signal <= 1.0
    assert 0.0 <= obs.feedback_volatility <= 1.0
    assert 0.0 <= obs.budget_remaining <= 1.0
    assert 0.0 <= obs.risk_tolerance <= 1.0
    assert 0.0 <= obs.latency_budget <= 1.0

    action = Action(
        recommended_item_id=obs.candidate_items[0].item_id,
        exploration_flag=False,
        confidence_score=0.8,
    )
    result = env.step(action)
    assert isinstance(result.reward, float)
    assert result.observation.turn_id == 1
    assert 0.0 <= result.observation.trust_signal <= 1.0
    assert 0.0 <= result.observation.budget_remaining <= 1.0


def test_task5_risk_collapse_requires_sustained_exposure():
    env = RecommendationPolicyEnvironment(seed=123)
    obs = env.reset("task_5", seed=123)
    high_risk_item = max(obs.candidate_items, key=lambda item: (item.risk, item.cost, item.latency))

    env.step(
        Action(
            recommended_item_id=high_risk_item.item_id,
            exploration_flag=False,
            confidence_score=0.95,
        )
    )
    assert env.hidden is not None
    assert env.hidden.risk_collapsed is False

    env = RecommendationPolicyEnvironment(seed=123)
    obs = env.reset("task_5", seed=123)
    env.hidden.risk_history = [0.35, 0.35]
    high_risk_item = max(obs.candidate_items, key=lambda item: (item.risk, item.cost, item.latency))

    env.step(
        Action(
            recommended_item_id=high_risk_item.item_id,
            exploration_flag=False,
            confidence_score=0.95,
        )
    )
    assert env.hidden.risk_collapsed is True
