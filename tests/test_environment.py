from app.simulator import RecommendationPolicyEnvironment
from app.models import Action
from app.tasks import get_task_config


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


def test_reset_initializes_hidden_state_from_runtime_task_config():
    env = RecommendationPolicyEnvironment(seed=123)
    env.reset("task_4", seed=456)
    assert env.hidden is not None
    assert env.task_cfg is not None
    assert env.hidden.regime == env.task_cfg.regime_init
    assert abs(env.hidden.latent_vol - env.task_cfg.latent_vol_init) < 1e-9
    assert abs(env.hidden.chi - env.task_cfg.chi_init) < 1e-9
    assert abs(env.hidden.budget_remaining - env.task_cfg.budget_init) < 1e-9


def test_reset_applies_light_user_heterogeneity_without_changing_task_identity():
    base_cfg = get_task_config("task_4")
    env = RecommendationPolicyEnvironment(seed=123)
    env.reset("task_4", seed=456)

    assert env.task_cfg is not None
    cfg = env.task_cfg

    assert cfg.task_id == base_cfg.task_id
    assert cfg.task_name == base_cfg.task_name
    assert cfg.memory_pref == base_cfg.memory_pref
    assert cfg.session_intent == base_cfg.session_intent
    assert cfg != base_cfg
    assert 0.70 <= cfg.lambda_F <= 0.99
    assert 0.05 <= cfg.delta_F <= 0.40
    assert 0.05 <= cfg.trust_gain <= 0.40
    assert 0.05 <= cfg.trust_loss <= 0.50
    assert 0.0 <= cfg.trust_volatility <= 0.30
    assert 0.0 <= cfg.latent_vol_init <= 1.0
    assert 0.60 <= cfg.regime_sticky <= 0.98
    assert 0.60 <= cfg.chi_init <= 0.95
    assert 0.70 <= cfg.budget_init <= 1.0
    assert 0.65 <= cfg.risk_tolerance_init <= 1.0
    assert 0.65 <= cfg.latency_budget_init <= 1.0


def test_task3_enters_non_stable_regime_under_conflict():
    env = RecommendationPolicyEnvironment(seed=123)
    obs = env.reset("task_3", seed=123)

    for _ in range(8):
        result = env.step(
            Action(
                recommended_item_id=obs.candidate_items[0].item_id,
                exploration_flag=False,
                confidence_score=0.82,
            )
        )
        obs = result.observation
        if result.done:
            break

    assert env.hidden is not None
    assert any(step.regime >= 1 for step in env.trajectory)
    assert 0.0 <= env.hidden.latent_vol <= 1.0


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
