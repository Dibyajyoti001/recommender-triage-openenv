from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List

import numpy as np

from .models import TaskSpec


# GLOBAL FIXED TOPIC BASIS
CATEGORY_NAMES: List[str] = [
    "Romance",      # 0
    "Comedy",       # 1
    "Drama",        # 2
    "Thriller",     # 3
    "Action",       # 4
    "Documentary",  # 5
    "Crime",        # 6
    "SciFi",        # 7
    "News",         # 8
    "Lifestyle",    # 9
]

K = 10
T_MAX = 20
REPETITION_WINDOW = 4


@dataclass(frozen=True)
class GlobalConstants:
    K: int = 10
    T_max: int = 20
    repetition_window: int = 4
    eta_q: float = 0.25
    w_c: float = 0.75
    w_i: float = 0.25
    lambda_i: float = 0.70
    delta_i: float = 0.12

    zeta1: float = 4.0
    zeta2: float = 2.5
    zeta3: float = 1.5
    zeta4: float = 1.15
    w_r: float = 0.60
    w_f: float = 0.20
    w_n: float = 0.10
    w_u: float = 0.05
    w_conf: float = 0.08

    beta_patience: float = 0.16

    # Calibrated for K=10 sparse-ish simplex geometry
    theta_chi: float = 0.14
    alpha_chi: float = 0.25
    chi_threshold_explore_penalty: float = 0.70

    theta_rec: float = 0.60
    lambda_A: float = 0.45
    early_stop_patience_floor_turns: int = 2


GLOBAL = GlobalConstants()


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    task_name: str
    difficulty: str
    description: str

    alpha: float
    lambda_c: float
    delta_c: float
    nu: float
    chi_init: float
    tau: float
    mu: float
    kappa: float
    epsilon_obs: float

    omega_1: float
    omega_2: float
    omega_3: float
    omega_4: float
    omega_trust: float
    omega_calibration: float

    memory_pref: List[float]
    session_intent: List[float]

    lambda_F: float
    delta_F: float
    lambda_H: float
    omega_mix: float
    conflict_strength: float
    trust_init: float = 0.72
    trust_sensitivity: float = 1.10
    trust_gain: float = 0.18
    trust_loss: float = 0.22
    trust_underclaim: float = 0.08
    trust_volatility: float = 0.12
    drift_trust_boost: float = 0.12
    confidence_reward_weight: float = GLOBAL.w_conf
    saturation_threshold: float = 1.10
    lambda_H_recovery: float = 0.0
    feedback_noise_std: float = 0.0
    dud_probability: float = 0.0
    dud_quality_range: List[float] = field(default_factory=lambda: [0.05, 0.25])
    volatility_window: int = 5
    calibration_decay_k: float = 8.0
    budget_init: float = 1.0
    risk_tolerance_init: float = 1.0
    latency_budget_init: float = 1.0
    budget_recovery: float = 0.03
    budget_spend_rate: float = 0.12
    latency_recovery: float = 0.05
    latency_spend_rate: float = 0.16
    risk_tolerance_recovery: float = 0.05
    risk_tolerance_decay: float = 0.12
    resource_pressure_loss: float = 0.08
    trust_collapse_threshold: float = -1.0
    risk_collapse_threshold: float = 2.0
    diversity_collapse_threshold: float = 2.0
    collapsed_satisfaction_cap: float = 0.62
    risk_noise_boost: float = 0.10
    collapse_volatility_floor: float = 0.06
    counterfactual_audit_weight: float = 0.0
    counterfactual_audit_horizon: int = 4
    counterfactual_audit_budget: int = 0
    counterfactual_fragility_threshold: float = 1.0
    regime_init: int = 0
    regime_sticky: float = 0.85
    latent_vol_init: float = 0.15
    latent_vol_revert: float = 0.92
    regime_drift_multiplier: List[float] = field(default_factory=lambda: [0.90, 1.10, 1.00, 1.25])
    regime_noise_multiplier: List[float] = field(default_factory=lambda: [0.65, 0.95, 1.05, 1.30])
    regime_fatigue_multiplier: List[float] = field(default_factory=lambda: [0.95, 1.05, 1.30, 1.15])
    regime_trust_loss_multiplier: List[float] = field(default_factory=lambda: [0.90, 1.00, 1.15, 1.35])

    def to_spec(self) -> TaskSpec:
        params = {
            **asdict(GLOBAL),
            "alpha": self.alpha,
            "lambda_c": self.lambda_c,
            "delta_c": self.delta_c,
            "nu": self.nu,
            "chi_init": self.chi_init,
            "tau": self.tau,
            "mu": self.mu,
            "kappa": self.kappa,
            "epsilon_obs": self.epsilon_obs,
            "lambda_F": self.lambda_F,
            "delta_F": self.delta_F,
            "lambda_H": self.lambda_H,
            "omega_mix": self.omega_mix,
            "conflict_strength": self.conflict_strength,
            "trust_init": self.trust_init,
            "trust_sensitivity": self.trust_sensitivity,
            "trust_gain": self.trust_gain,
            "trust_loss": self.trust_loss,
            "trust_underclaim": self.trust_underclaim,
            "trust_volatility": self.trust_volatility,
            "drift_trust_boost": self.drift_trust_boost,
            "confidence_reward_weight": self.confidence_reward_weight,
            "saturation_threshold": self.saturation_threshold,
            "lambda_H_recovery": self.lambda_H_recovery,
            "feedback_noise_std": self.feedback_noise_std,
            "dud_probability": self.dud_probability,
            "dud_quality_range": self.dud_quality_range,
            "volatility_window": self.volatility_window,
            "calibration_decay_k": self.calibration_decay_k,
            "budget_init": self.budget_init,
            "risk_tolerance_init": self.risk_tolerance_init,
            "latency_budget_init": self.latency_budget_init,
            "budget_recovery": self.budget_recovery,
            "budget_spend_rate": self.budget_spend_rate,
            "latency_recovery": self.latency_recovery,
            "latency_spend_rate": self.latency_spend_rate,
            "risk_tolerance_recovery": self.risk_tolerance_recovery,
            "risk_tolerance_decay": self.risk_tolerance_decay,
            "resource_pressure_loss": self.resource_pressure_loss,
            "trust_collapse_threshold": self.trust_collapse_threshold,
            "risk_collapse_threshold": self.risk_collapse_threshold,
            "diversity_collapse_threshold": self.diversity_collapse_threshold,
            "collapsed_satisfaction_cap": self.collapsed_satisfaction_cap,
            "risk_noise_boost": self.risk_noise_boost,
            "collapse_volatility_floor": self.collapse_volatility_floor,
            "counterfactual_audit_weight": self.counterfactual_audit_weight,
            "counterfactual_audit_horizon": self.counterfactual_audit_horizon,
            "counterfactual_audit_budget": self.counterfactual_audit_budget,
            "counterfactual_fragility_threshold": self.counterfactual_fragility_threshold,
            "regime_init": self.regime_init,
            "regime_sticky": self.regime_sticky,
            "latent_vol_init": self.latent_vol_init,
            "latent_vol_revert": self.latent_vol_revert,
            "regime_drift_multiplier": self.regime_drift_multiplier,
            "regime_noise_multiplier": self.regime_noise_multiplier,
            "regime_fatigue_multiplier": self.regime_fatigue_multiplier,
            "regime_trust_loss_multiplier": self.regime_trust_loss_multiplier,
            "topic_names": CATEGORY_NAMES,
        }
        return TaskSpec(
            task_id=self.task_id,
            task_name=self.task_name,
            difficulty=self.difficulty,
            description=self.description,
            max_turns=GLOBAL.T_max,
            grader_weights={
                "satisfaction": self.omega_1,
                "diversity": self.omega_2,
                "adaptation": self.omega_3,
                "memory_use": self.omega_4,
                "trust": self.omega_trust,
                "calibration": self.omega_calibration,
            },
            parameters=params,
        )


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "task_1": TaskConfig(
        task_id="task_1",
        task_name="Stable Preference Exploitation",
        difficulty="easy",
        description=(
            "Historical memory is reliable, live intent mostly matches memory, fatigue grows slowly, "
            "and distractors are weaker."
        ),
        alpha=0.45,
        lambda_c=0.80,
        delta_c=0.12,
        nu=0.68,
        chi_init=0.85,
        tau=0.20,
        mu=0.08,
        kappa=0.18,
        epsilon_obs=0.05,
        omega_1=0.60,
        omega_2=0.20,
        omega_3=0.00,
        omega_4=0.20,
        omega_trust=0.00,
        omega_calibration=0.00,
        memory_pref=[0.36, 0.22, 0.14, 0.06, 0.03, 0.08, 0.03, 0.02, 0.03, 0.03],
        session_intent=[0.32, 0.24, 0.16, 0.06, 0.03, 0.07, 0.03, 0.02, 0.03, 0.04],
        lambda_F=0.82,
        delta_F=0.12,
        lambda_H=0.86,
        omega_mix=0.22,
        conflict_strength=0.18,
        trust_init=0.78,
        trust_sensitivity=0.95,
        trust_gain=0.16,
        trust_loss=0.18,
        trust_underclaim=0.05,
        trust_volatility=0.04,
        drift_trust_boost=0.06,
        budget_init=0.96,
        risk_tolerance_init=0.94,
        latency_budget_init=0.95,
        budget_recovery=0.05,
        budget_spend_rate=0.08,
        latency_recovery=0.07,
        latency_spend_rate=0.10,
        risk_tolerance_recovery=0.07,
        risk_tolerance_decay=0.08,
        resource_pressure_loss=0.02,
        regime_init=0,
        regime_sticky=0.92,
        latent_vol_init=0.08,
        latent_vol_revert=0.94,
        regime_drift_multiplier=[0.88, 1.02, 0.95, 1.10],
        regime_noise_multiplier=[0.45, 0.75, 0.90, 1.10],
        regime_fatigue_multiplier=[0.90, 1.00, 1.12, 1.08],
        regime_trust_loss_multiplier=[0.88, 0.96, 1.06, 1.18],
    ),
    "task_2": TaskConfig(
        task_id="task_2",
        task_name="Repetition Fatigue Control",
        difficulty="medium",
        description=(
            "The user strongly prefers a narrow band of topics, but repetition fatigue builds quickly "
            "and novelty tolerance is lower."
        ),
        alpha=0.55,
        lambda_c=0.88,
        delta_c=0.22,
        nu=0.42,
        chi_init=0.80,
        tau=0.18,
        mu=0.06,
        kappa=0.14,
        epsilon_obs=0.07,
        omega_1=0.40,
        omega_2=0.40,
        omega_3=0.00,
        omega_4=0.20,
        omega_trust=0.00,
        omega_calibration=0.00,
        memory_pref=[0.48, 0.24, 0.08, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03, 0.02],
        session_intent=[0.44, 0.28, 0.08, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03, 0.02],
        lambda_F=0.90,
        delta_F=0.22,
        lambda_H=0.90,
        omega_mix=0.24,
        conflict_strength=0.30,
        trust_init=0.74,
        trust_sensitivity=1.05,
        trust_gain=0.16,
        trust_loss=0.22,
        trust_underclaim=0.06,
        trust_volatility=0.06,
        drift_trust_boost=0.08,
        budget_init=0.90,
        risk_tolerance_init=0.88,
        latency_budget_init=0.88,
        budget_recovery=0.04,
        budget_spend_rate=0.10,
        latency_recovery=0.05,
        latency_spend_rate=0.12,
        risk_tolerance_recovery=0.06,
        risk_tolerance_decay=0.10,
        resource_pressure_loss=0.04,
        regime_init=0,
        regime_sticky=0.88,
        latent_vol_init=0.22,
        latent_vol_revert=0.90,
        regime_drift_multiplier=[0.92, 1.05, 1.02, 1.16],
        regime_noise_multiplier=[0.55, 0.88, 1.02, 1.18],
        regime_fatigue_multiplier=[1.00, 1.08, 1.42, 1.24],
        regime_trust_loss_multiplier=[0.92, 1.00, 1.14, 1.28],
    ),
    "task_3": TaskConfig(
        task_id="task_3",
        task_name="Memory vs Live Signal Conflict",
        difficulty="hard",
        description=(
            "Memory points toward one mixed-topic region, but the live session shifts toward another. "
            "The agent must override stale memory and recover relevance."
        ),
        alpha=0.82,
        lambda_c=0.85,
        delta_c=0.18,
        nu=0.48,
        chi_init=0.90,
        tau=0.35,
        mu=0.05,
        kappa=0.40,
        epsilon_obs=0.12,
        omega_1=0.30,
        omega_2=0.20,
        omega_3=0.30,
        omega_4=0.20,
        omega_trust=0.00,
        omega_calibration=0.00,
        memory_pref=[0.04, 0.06, 0.08, 0.10, 0.04, 0.26, 0.18, 0.04, 0.14, 0.06],
        session_intent=[0.28, 0.30, 0.18, 0.06, 0.02, 0.04, 0.02, 0.02, 0.04, 0.04],
        lambda_F=0.86,
        delta_F=0.18,
        lambda_H=0.88,
        omega_mix=0.16,
        conflict_strength=0.82,
        trust_init=0.72,
        trust_sensitivity=1.10,
        trust_gain=0.14,
        trust_loss=0.26,
        trust_underclaim=0.08,
        trust_volatility=0.08,
        drift_trust_boost=0.14,
        budget_init=0.86,
        risk_tolerance_init=0.82,
        latency_budget_init=0.84,
        budget_recovery=0.04,
        budget_spend_rate=0.11,
        latency_recovery=0.05,
        latency_spend_rate=0.13,
        risk_tolerance_recovery=0.05,
        risk_tolerance_decay=0.11,
        resource_pressure_loss=0.05,
        regime_init=0,
        regime_sticky=0.80,
        latent_vol_init=0.18,
        latent_vol_revert=0.88,
        regime_drift_multiplier=[0.90, 1.24, 1.08, 1.32],
        regime_noise_multiplier=[0.60, 0.96, 1.08, 1.28],
        regime_fatigue_multiplier=[0.96, 1.08, 1.24, 1.18],
        regime_trust_loss_multiplier=[0.92, 1.04, 1.16, 1.30],
    ),
    "task_4": TaskConfig(
        task_id="task_4",
        task_name="Echo Chamber Collapse",
        difficulty="hard",
        description=(
            "Early recommendations look excellent, but over-serving one narrow topic silently saturates slow-history "
            "pressure, erodes trust, and eventually collapses the session. The agent must diversify before the crash "
            "is visible in raw satisfaction."
        ),
        alpha=0.52,
        lambda_c=0.90,
        delta_c=0.24,
        nu=0.32,
        chi_init=0.88,
        tau=0.18,
        mu=0.05,
        kappa=0.24,
        epsilon_obs=0.05,
        omega_1=0.25,
        omega_2=0.30,
        omega_3=0.20,
        omega_4=0.00,
        omega_trust=0.25,
        omega_calibration=0.00,
        memory_pref=[0.02, 0.03, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.55, 0.07],
        session_intent=[0.02, 0.03, 0.04, 0.04, 0.03, 0.08, 0.04, 0.07, 0.58, 0.07],
        lambda_F=0.92,
        delta_F=0.24,
        lambda_H=0.94,
        omega_mix=0.18,
        conflict_strength=0.20,
        trust_init=0.82,
        trust_sensitivity=1.40,
        trust_gain=0.18,
        trust_loss=0.28,
        trust_underclaim=0.06,
        trust_volatility=0.05,
        drift_trust_boost=0.12,
        confidence_reward_weight=0.10,
        saturation_threshold=0.48,
        lambda_H_recovery=0.985,
        budget_init=0.78,
        risk_tolerance_init=0.72,
        latency_budget_init=0.76,
        budget_recovery=0.03,
        budget_spend_rate=0.14,
        latency_recovery=0.04,
        latency_spend_rate=0.18,
        risk_tolerance_recovery=0.04,
        risk_tolerance_decay=0.14,
        resource_pressure_loss=0.09,
        trust_collapse_threshold=0.22,
        risk_collapse_threshold=0.28,
        diversity_collapse_threshold=0.30,
        collapsed_satisfaction_cap=0.62,
        risk_noise_boost=0.08,
        collapse_volatility_floor=0.05,
        counterfactual_audit_weight=0.12,
        counterfactual_audit_horizon=4,
        counterfactual_audit_budget=2,
        counterfactual_fragility_threshold=0.48,
        # === REGIME TUNING — this keeps the engine fully active but fair ===
        regime_init=0,                    
        regime_sticky=0.82,               
        latent_vol_init=0.28,
        latent_vol_revert=0.84,
        regime_drift_multiplier=[0.92, 1.14, 1.10, 1.34],
        regime_noise_multiplier=[0.62, 0.98, 1.08, 1.32],
        regime_fatigue_multiplier=[0.98, 1.05, 1.32, 1.22],   
        regime_trust_loss_multiplier=[0.94, 1.02, 1.16, 1.32],
    ),
    "task_5": TaskConfig(
        task_id="task_5",
        task_name="Noisy Signal Calibration",
        difficulty="hard",
        description=(
            "The agent must manage user trust when feedback is noisy and aligned-looking candidates sometimes disappoint. "
            "Calibration matters as much as relevance."
        ),
        alpha=0.70,
        lambda_c=0.84,
        delta_c=0.16,
        nu=0.46,
        chi_init=0.76,
        tau=0.24,
        mu=0.08,
        kappa=0.30,
        epsilon_obs=0.14,
        omega_1=0.25,
        omega_2=0.00,
        omega_3=0.00,
        omega_4=0.10,
        omega_trust=0.20,
        omega_calibration=0.45,
        memory_pref=[0.12, 0.16, 0.09, 0.08, 0.07, 0.10, 0.09, 0.11, 0.10, 0.08],
        session_intent=[0.10, 0.18, 0.10, 0.08, 0.08, 0.08, 0.08, 0.12, 0.10, 0.08],
        lambda_F=0.86,
        delta_F=0.18,
        lambda_H=0.88,
        omega_mix=0.22,
        conflict_strength=0.38,
        trust_init=0.70,
        trust_sensitivity=1.15,
        trust_gain=0.20,
        trust_loss=0.28,
        trust_underclaim=0.10,
        trust_volatility=0.16,
        drift_trust_boost=0.18,
        confidence_reward_weight=0.20,
        feedback_noise_std=0.20,
        dud_probability=0.25,
        dud_quality_range=[0.05, 0.25],
        volatility_window=5,
        calibration_decay_k=8.0,
        budget_init=0.80,
        risk_tolerance_init=0.74,
        latency_budget_init=0.78,
        budget_recovery=0.03,
        budget_spend_rate=0.13,
        latency_recovery=0.04,
        latency_spend_rate=0.16,
        risk_tolerance_recovery=0.04,
        risk_tolerance_decay=0.14,
        resource_pressure_loss=0.06,
        trust_collapse_threshold=0.18,
        risk_collapse_threshold=0.26,
        diversity_collapse_threshold=0.34,
        collapsed_satisfaction_cap=0.66,
        risk_noise_boost=0.08,
        collapse_volatility_floor=0.06,
        counterfactual_audit_weight=0.12,
        counterfactual_audit_horizon=4,
        counterfactual_audit_budget=2,
        counterfactual_fragility_threshold=0.26,
        regime_init=0,
        regime_sticky=0.78,
        latent_vol_init=0.45,
        latent_vol_revert=0.80,
        regime_drift_multiplier=[0.94, 1.12, 1.06, 1.24],
        regime_noise_multiplier=[0.75, 1.10, 1.18, 1.40],
        regime_fatigue_multiplier=[0.96, 1.08, 1.20, 1.14],
        regime_trust_loss_multiplier=[0.96, 1.06, 1.16, 1.34],
    ),
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _light_user_heterogeneity(base_cfg: TaskConfig, rng: np.random.Generator) -> TaskConfig:
    """
    Apply small bounded per-episode noise to user-dynamic parameters only.
    Task identity, topic basis, grader weights, and task semantics stay fixed.
    """

    def mul(scale: float) -> float:
        return 1.0 + float(rng.uniform(-scale, scale))

    return replace(
        base_cfg,
        lambda_F=_clip(base_cfg.lambda_F * mul(0.08), 0.70, 0.99),
        delta_F=_clip(base_cfg.delta_F * mul(0.12), 0.05, 0.40),
        trust_gain=_clip(base_cfg.trust_gain * mul(0.12), 0.05, 0.40),
        trust_loss=_clip(base_cfg.trust_loss * mul(0.12), 0.05, 0.50),
        trust_volatility=_clip(base_cfg.trust_volatility * mul(0.15), 0.0, 0.30),
        latent_vol_init=_clip(base_cfg.latent_vol_init * mul(0.12), 0.0, 1.0),
        regime_sticky=_clip(base_cfg.regime_sticky * mul(0.06), 0.60, 0.98),
        chi_init=_clip(base_cfg.chi_init * mul(0.08), 0.60, 0.95),
        budget_init=_clip(base_cfg.budget_init * mul(0.05), 0.70, 1.0),
        risk_tolerance_init=_clip(base_cfg.risk_tolerance_init * mul(0.05), 0.65, 1.0),
        latency_budget_init=_clip(base_cfg.latency_budget_init * mul(0.05), 0.65, 1.0),
    )


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_id={task_id!r}")
    return TASK_CONFIGS[task_id]


def list_task_specs() -> List[TaskSpec]:
    return [cfg.to_spec() for cfg in TASK_CONFIGS.values()]
