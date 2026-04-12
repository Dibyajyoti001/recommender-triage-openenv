from app.tasks import TASK_CONFIGS, get_task_config


def test_exact_five_tasks_present():
    assert set(TASK_CONFIGS.keys()) == {"task_1", "task_2", "task_3", "task_4", "task_5"}


def test_task_weights_sum_to_one():
    for task_id in TASK_CONFIGS:
        cfg = get_task_config(task_id)
        total = cfg.omega_1 + cfg.omega_2 + cfg.omega_3 + cfg.omega_4 + cfg.omega_trust + cfg.omega_calibration
        assert abs(total - 1.0) < 1e-9


def test_regime_tables_are_well_formed():
    for task_id in TASK_CONFIGS:
        cfg = get_task_config(task_id)
        assert 0 <= cfg.regime_init <= 3
        assert 0.0 <= cfg.latent_vol_init <= 1.0
        assert 0.0 <= cfg.latent_vol_revert <= 1.0
        for name in (
            "regime_drift_multiplier",
            "regime_noise_multiplier",
            "regime_fatigue_multiplier",
            "regime_trust_loss_multiplier",
        ):
            values = getattr(cfg, name)
            assert len(values) == 4
            assert all(v >= 0.0 for v in values)
