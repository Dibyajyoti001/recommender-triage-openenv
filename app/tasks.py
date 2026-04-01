from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from .models import TaskSpec


CATEGORY_NAMES: List[str] = [
    "Documentary",
    "Comedy",
    "Thriller",
    "News",
    "Lifestyle",
]

K = 5
T_MAX = 20
REPETITION_WINDOW = 4


@dataclass(frozen=True)
class GlobalConstants:
    K: int = 5
    T_max: int = 20
    repetition_window: int = 4
    eta_q: float = 0.25
    w_c: float = 0.75
    w_i: float = 0.25
    lambda_i: float = 0.70
    delta_i: float = 0.12
    beta_good: float = 0.10
    beta_bad: float = 0.18
    gamma1: float = 0.50
    gamma2: float = 0.30
    gamma3: float = 0.20
    zeta1: float = 4.0
    zeta2: float = 2.5
    zeta3: float = 1.5
    w_r: float = 0.60
    w_f: float = 0.20
    w_n: float = 0.10
    w_u: float = 0.05
    w_conf: float = 0.05
    theta_rec: float = 0.60
    lambda_A: float = 0.45
    eta_chi: float = 0.12
    beta_chi: float = 0.05
    chi_threshold_explore_penalty: float = 0.70
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

    # task-specific grader weights
    omega_1: float
    omega_2: float
    omega_3: float
    omega_4: float

    # fixed initial preference / intent templates
    memory_pref: List[float]
    session_intent: List[float]

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
        }
        return TaskSpec(
            task_id=self.task_id,
            task_name=self.task_name,
            difficulty=self.difficulty,  # type: ignore[arg-type]
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
            "Historical memory is reliable, live intent matches memory, fatigue grows slowly, "
            "and distractors are weaker. The agent should exploit without wasting steps."
        ),
        alpha=0.40,
        lambda_c=0.80,
        delta_c=0.12,
        nu=0.65,
        chi_init=0.85,
        tau=0.20,
        mu=0.08,
        kappa=0.18,
        epsilon_obs=0.05,
        omega_1=0.60,
        omega_2=0.20,
        omega_3=0.00,
        omega_4=0.20,
        memory_pref=[0.46, 0.16, 0.14, 0.14, 0.10],
        session_intent=[0.43, 0.17, 0.15, 0.15, 0.10],
    ),
    "task_2": TaskConfig(
        task_id="task_2",
        task_name="Repetition Fatigue Control",
        difficulty="medium",
        description=(
            "The user strongly prefers one category, but repetition fatigue is strong and novelty "
            "tolerance is lower. The agent must trade off relevance and boredom."
        ),
        alpha=0.50,
        lambda_c=0.88,
        delta_c=0.22,
        nu=0.35,
        chi_init=0.80,
        tau=0.18,
        mu=0.06,
        kappa=0.14,
        epsilon_obs=0.07,
        omega_1=0.40,
        omega_2=0.40,
        omega_3=0.00,
        omega_4=0.20,
        memory_pref=[0.58, 0.10, 0.12, 0.10, 0.10],
        session_intent=[0.55, 0.11, 0.12, 0.11, 0.11],
    ),
    "task_3": TaskConfig(
        task_id="task_3",
        task_name="Memory vs Live Signal Conflict",
        difficulty="hard",
        description=(
            "Memory strongly points to one category, but live session behavior shifts toward another. "
            "The agent must detect conflict, override stale memory, and recover quickly."
        ),
        alpha=0.80,
        lambda_c=0.85,
        delta_c=0.18,
        nu=0.45,
        chi_init=0.90,
        tau=0.35,
        mu=0.05,
        kappa=0.35,
        epsilon_obs=0.12,
        omega_1=0.30,
        omega_2=0.20,
        omega_3=0.30,
        omega_4=0.20,
        memory_pref=[0.56, 0.08, 0.12, 0.14, 0.10],   # documentary-heavy
        session_intent=[0.12, 0.52, 0.12, 0.12, 0.12],  # comedy-heavy
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_id={task_id!r}")
    return TASK_CONFIGS[task_id]


def list_task_specs() -> List[TaskSpec]:
    return [cfg.to_spec() for cfg in TASK_CONFIGS.values()]