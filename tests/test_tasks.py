from app.tasks import TASK_CONFIGS, get_task_config


def test_exact_three_tasks_present():
    assert set(TASK_CONFIGS.keys()) == {"task_1", "task_2", "task_3"}


def test_task_weights_sum_to_one():
    for task_id in TASK_CONFIGS:
        cfg = get_task_config(task_id)
        total = cfg.omega_1 + cfg.omega_2 + cfg.omega_3 + cfg.omega_4
        assert abs(total - 1.0) < 1e-9