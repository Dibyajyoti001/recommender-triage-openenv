from app.simulator import RecommendationPolicyEnvironment
from app.models import Action


def test_environment_reset_and_step():
    env = RecommendationPolicyEnvironment(seed=123)
    obs = env.reset("task_1", seed=456)
    assert obs.task_id == "task_1"
    assert len(obs.candidate_items) == 6

    action = Action(
        recommended_item_id=obs.candidate_items[0].item_id,
        exploration_flag=False,
        confidence_score=0.8,
    )
    result = env.step(action)
    assert isinstance(result.reward, float)
    assert result.observation.turn_id == 1