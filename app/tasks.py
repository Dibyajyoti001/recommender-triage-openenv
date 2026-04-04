from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

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

    memory_pref: List[float]
    session_intent: List[float]

    lambda_F: float
    delta_F: float
    lambda_H: float
    omega_mix: float
    conflict_strength: float

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
        memory_pref=[0.36, 0.22, 0.14, 0.06, 0.03, 0.08, 0.03, 0.02, 0.03, 0.03],
        session_intent=[0.32, 0.24, 0.16, 0.06, 0.03, 0.07, 0.03, 0.02, 0.03, 0.04],
        lambda_F=0.82,
        delta_F=0.12,
        lambda_H=0.86,
        omega_mix=0.22,
        conflict_strength=0.18,
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
        memory_pref=[0.48, 0.24, 0.08, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03, 0.02],
        session_intent=[0.44, 0.28, 0.08, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03, 0.02],
        lambda_F=0.90,
        delta_F=0.22,
        lambda_H=0.90,
        omega_mix=0.24,
        conflict_strength=0.30,
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
        memory_pref=[0.04, 0.06, 0.08, 0.10, 0.04, 0.26, 0.18, 0.04, 0.14, 0.06],
        session_intent=[0.28, 0.30, 0.18, 0.06, 0.02, 0.04, 0.02, 0.02, 0.04, 0.04],
        lambda_F=0.86,
        delta_F=0.18,
        lambda_H=0.88,
        omega_mix=0.16,
        conflict_strength=0.82,
    ),
}

def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_id={task_id!r}")
    return TASK_CONFIGS[task_id]


def list_task_specs() -> List[TaskSpec]:
    return [cfg.to_spec() for cfg in TASK_CONFIGS.values()]