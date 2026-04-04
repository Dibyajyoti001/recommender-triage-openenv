from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .candidate_pool import build_candidate_pool
from .models import (
    Action,
    EnvironmentState,
    FinalGradeBreakdown,
    HiddenState,
    MemorySummary,
    Observation,
    RecentInteraction,
    StepResult,
)
from .reward import (
    compute_step_reward,
    confidence_bucket,
    feedback_bucket,
    pressure_bucket,
    update_patience,
)
from .tasks import CATEGORY_NAMES, GLOBAL, K, TaskConfig, get_task_config
from .graders import final_grade , TrajectoryStep


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.maximum(arr, 0.0)
    s = float(arr.sum())
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


@dataclass
class TrajectoryStep:
    turn_id: int
    chosen_item_id: int
    chosen_category_id: int
    chosen_category_name: str
    relevance: float
    satisfaction_proxy: float
    memory_confidence: float
    m: List[float]
    z: List[float]
    p_before: float
    p_after: float
    exploration_flag: bool
    confidence_score: float


class RecommendationPolicyEnvironment:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.hidden: Optional[HiddenState] = None
        self.task_cfg: Optional[TaskConfig] = None
        self.candidate_items = []
        self.recent_interactions: List[RecentInteraction] = []
        self.trajectory: List[TrajectoryStep] = []
        self.latest_reward_breakdown = None
        self.final_breakdown: Optional[FinalGradeBreakdown] = None
        self._zero_patience_streak = 0

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        self.task_cfg = get_task_config(task_id)
        self.recent_interactions = []
        self.trajectory = []
        self.latest_reward_breakdown = None
        self.final_breakdown = None
        self._zero_patience_streak = 0

        m = _normalize(np.asarray(self.task_cfg.memory_pref, dtype=float))
        z = _normalize(np.asarray(self.task_cfg.session_intent, dtype=float))

        h0 = np.ones(K, dtype=float) / K
        f0 = np.zeros(K, dtype=float)

        self.hidden = HiddenState(
            task_id=task_id,
            turn=0,
            m=[float(x) for x in m],
            z=[float(x) for x in z],
            H_topic=[float(x) for x in h0],
            F_topic=[float(x) for x in f0],
            history_topic_vectors=[],
            F_cat=[0.0 for _ in range(K)],
            F_item={},
            nu=float(self.task_cfg.nu),
            p=1.0,
            chi=float(self.task_cfg.chi_init),
            history_item_ids=[],
            history_category_ids=[],
            drift_turn=None,
            recovered_turn=None,
            last_feedback_bucket="neutral",
            rng_seed=self.seed,
        )

        self.candidate_items = build_candidate_pool(
            task_id=task_id,
            turn=0,
            alpha=self.task_cfg.alpha,
            eta_q=GLOBAL.eta_q,
            z=self.hidden.z,
            m=self.hidden.m,
            item_fatigue=self.hidden.F_item,
            history_categories=self.hidden.history_category_ids,
            rng=self.rng,
            H_topic=self.hidden.H_topic,
            F_topic=self.hidden.F_topic,
        )
        return self._build_observation(session_feedback_signal=0.5)

    def state(self) -> EnvironmentState:
        if self.hidden is None or self.task_cfg is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return EnvironmentState(
            hidden_state=self.hidden,
            task_spec=self.task_cfg.to_spec(),
            trajectory_length=len(self.trajectory),
            latest_reward_breakdown=self.latest_reward_breakdown,
        )

    def _memory_summary(self) -> MemorySummary:
        assert self.hidden is not None
        top_idxs = sorted(range(K), key=lambda i: self.hidden.m[i], reverse=True)[:3]
        return MemorySummary(
            top_categories=top_idxs,
            top_category_names=[CATEGORY_NAMES[i] for i in top_idxs],
            coarse_scores=[float(round(self.hidden.m[i], 4)) for i in top_idxs],
            summary_text=(
                f"Historical preference appears strongest for "
                f"{CATEGORY_NAMES[top_idxs[0]]}, then {CATEGORY_NAMES[top_idxs[1]]}, "
                f"then {CATEGORY_NAMES[top_idxs[2]]}."
            ),
        )

    def _build_observation(self, session_feedback_signal: float) -> Observation:
        assert self.hidden is not None

        repetition_counts = [self.hidden.history_category_ids.count(i) for i in range(K)]
        rho_display = _dot(self.hidden.H_topic, self.hidden.z)

        return Observation(
            task_id=self.hidden.task_id,
            task_name=get_task_config(self.hidden.task_id).task_name,
            turn_id=self.hidden.turn,
            max_turns=GLOBAL.T_max,
            memory_summary=self._memory_summary(),
            recent_interactions=self.recent_interactions[-5:],
            candidate_items=self.candidate_items,
            repetition_counts=repetition_counts,
            repetition_pressure_bucket=pressure_bucket(rho_display),
            memory_confidence_bucket=confidence_bucket(self.hidden.chi),
            memory_confidence=float(round(self.hidden.chi, 6)),
            session_feedback_signal=float(round(session_feedback_signal, 6)),
            done_hint="Session continues until max turns or patience collapses repeatedly.",
        )

    def _drift_targets(self, current_z: np.ndarray, current_m: np.ndarray, task_id: str) -> Tuple[np.ndarray, np.ndarray]:
        assert self.task_cfg is not None
        phi_t = _normalize((1.0 - self.task_cfg.omega_mix) * current_z + self.task_cfg.omega_mix * current_m)

        u = np.ones(K, dtype=float) / K
        anti_m = _normalize(1.0 - current_m)

        if task_id == "task_3":
            s = self.task_cfg.conflict_strength
            psi_t = _normalize((1.0 - s) * u + s * anti_m)
        else:
            s = min(0.55, self.task_cfg.conflict_strength)
            anti_z = _normalize(1.0 - current_z)
            psi_t = _normalize((1.0 - s) * anti_z + s * anti_m)

        return phi_t, psi_t

    def step(self, action: Action) -> StepResult:
        if self.hidden is None or self.task_cfg is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self.hidden.turn >= GLOBAL.T_max:
            raise RuntimeError("Episode already terminated. Call reset() for a new one.")

        item_lookup = {item.item_id: item for item in self.candidate_items}
        chosen = item_lookup.get(action.recommended_item_id, self.candidate_items[0])

        pre_m = list(self.hidden.m)
        pre_z = list(self.hidden.z)
        pre_H = list(self.hidden.H_topic)
        pre_F = list(self.hidden.F_topic)
        pre_chi = float(self.hidden.chi)
        pre_p = float(self.hidden.p)
        pre_argmax_z = int(np.argmax(np.asarray(pre_z)))

        chosen_x = list(
            chosen.topic_vector
            or chosen.metadata.get("topic_vector")
            or [1.0 if i == chosen.category_id else 0.0 for i in range(K)]
        )

        category_fatigue_val = float(self.hidden.F_cat[chosen.category_id])
        item_fatigue_val = float(self.hidden.F_item.get(chosen.item_id, 0.0))

        reward_value, reward_breakdown, aux = compute_step_reward(
            alpha=self.task_cfg.alpha,
            eta_q=GLOBAL.eta_q,
            z=pre_z,
            m=pre_m,
            item=chosen,
            category_fatigue_value=category_fatigue_val,
            item_fatigue_value=item_fatigue_val,
            history_categories=self.hidden.history_category_ids,
            nu=self.hidden.nu,
            chi_t=pre_chi,
            exploration_flag=action.exploration_flag,
            confidence_score=action.confidence_score,
            repetition_window=GLOBAL.repetition_window,
            w_c=GLOBAL.w_c,
            w_i=GLOBAL.w_i,
            H_topic=pre_H,
            F_topic=pre_F,
        )
        self.latest_reward_breakdown = reward_breakdown

        new_F_topic = [
            float(self.task_cfg.lambda_F * pre_F[i] + self.task_cfg.delta_F * chosen_x[i])
            for i in range(K)
        ]

        new_H_topic = [
            float(self.task_cfg.lambda_H * pre_H[i] + (1.0 - self.task_cfg.lambda_H) * chosen_x[i])
            for i in range(K)
        ]
        new_H_topic = list(_normalize(np.asarray(new_H_topic, dtype=float)))

        new_F_cat = [float(x) for x in new_F_topic]

        new_F_item = dict(self.hidden.F_item)
        for item_id, current_val in list(new_F_item.items()):
            chosen_now = item_id == chosen.item_id
            updated = GLOBAL.lambda_i * current_val + (GLOBAL.delta_i if chosen_now else 0.0)
            if updated <= 1e-8:
                new_F_item.pop(item_id, None)
            else:
                new_F_item[item_id] = float(updated)
        if chosen.item_id not in new_F_item:
            new_F_item[chosen.item_id] = float(GLOBAL.delta_i)

        y_t = float(aux["satisfaction_proxy"])
        new_p = update_patience(pre_p, y_t)

        z_arr = np.asarray(pre_z, dtype=float)
        m_arr = np.asarray(pre_m, dtype=float)
        phi_t, psi_t = self._drift_targets(z_arr, m_arr, self.hidden.task_id)

        # FIXED:
        # - Task 3 gets one forced real conflict event at turn 4.
        # - Otherwise: use phi_t when patience is healthy, psi_t when patience is low.
        if self.hidden.task_id == "task_3" and self.hidden.turn == 4 and self.hidden.drift_turn is None:
            new_z = _normalize(psi_t)
            self.hidden.drift_turn = self.hidden.turn
        elif new_p > self.task_cfg.tau:
            new_z = (1.0 - self.task_cfg.mu) * z_arr + self.task_cfg.mu * phi_t
            new_z = _normalize(new_z)
        else:
            new_z = (1.0 - self.task_cfg.kappa) * z_arr + self.task_cfg.kappa * psi_t
            new_z = _normalize(new_z)

        post_argmax_z = int(np.argmax(new_z))
        drift_happened = (post_argmax_z != pre_argmax_z)
        if drift_happened and self.hidden.drift_turn is None:
            self.hidden.drift_turn = self.hidden.turn

        alignment = float(np.dot(new_z, m_arr))
        new_chi = float(np.clip(pre_chi + GLOBAL.alpha_chi * (alignment - GLOBAL.theta_chi), 0.0, 1.0))

        self.trajectory.append(
            TrajectoryStep(
                turn_id=self.hidden.turn,
                chosen_item_id=chosen.item_id,
                chosen_category_id=chosen.category_id,
                chosen_category_name=chosen.category_name,
                relevance=float(aux["relevance"]),
                satisfaction_proxy=float(aux["satisfaction_proxy"]),
                memory_confidence=pre_chi,
                m=list(pre_m),
                z=list(pre_z),
                p_before=float(pre_p),
                p_after=float(new_p),
                exploration_flag=action.exploration_flag,
                confidence_score=action.confidence_score,
            )
        )

        self.hidden.history_item_ids.append(chosen.item_id)
        self.hidden.history_category_ids.append(chosen.category_id)
        self.hidden.history_topic_vectors.append([float(x) for x in chosen_x])

        fb_bucket = feedback_bucket(aux["satisfaction_proxy"])
        self.recent_interactions.append(
            RecentInteraction(
                turn_id=self.hidden.turn,
                item_id=chosen.item_id,
                category_id=chosen.category_id,
                category_name=chosen.category_name,
                confidence_score=action.confidence_score,
                exploration_flag=action.exploration_flag,
                reward=reward_value,
                satisfaction_proxy=float(aux["satisfaction_proxy"]),
                feedback_bucket=fb_bucket,
            )
        )

        self.hidden.turn += 1
        self.hidden.m = list(pre_m)
        self.hidden.z = [float(x) for x in new_z]
        self.hidden.H_topic = [float(x) for x in new_H_topic]
        self.hidden.F_topic = [float(x) for x in new_F_topic]
        self.hidden.F_cat = [float(x) for x in new_F_cat]
        self.hidden.F_item = {int(k): float(v) for k, v in new_F_item.items()}
        self.hidden.p = float(new_p)
        self.hidden.chi = float(new_chi)
        self.hidden.last_feedback_bucket = fb_bucket

        if self.hidden.p <= 1e-8:
            self._zero_patience_streak += 1
        else:
            self._zero_patience_streak = 0

        done = (
            self.hidden.turn >= GLOBAL.T_max
            or self._zero_patience_streak >= GLOBAL.early_stop_patience_floor_turns
        )

        info: Dict[str, Any] = {
            "task_id": self.hidden.task_id,
            "task_name": self.task_cfg.task_name,
            "reward_breakdown": reward_breakdown.model_dump(),
        }

        if not done:
            self.candidate_items = build_candidate_pool(
                task_id=self.hidden.task_id,
                turn=self.hidden.turn,
                alpha=self.task_cfg.alpha,
                eta_q=GLOBAL.eta_q,
                z=self.hidden.z,
                m=self.hidden.m,
                item_fatigue=self.hidden.F_item,
                history_categories=self.hidden.history_category_ids,
                rng=self.rng,
                H_topic=self.hidden.H_topic,
                F_topic=self.hidden.F_topic,
            )
            obs = self._build_observation(session_feedback_signal=float(aux["satisfaction_proxy"]))
        else:
            self.candidate_items = []
            self.final_breakdown = final_grade(self.trajectory, self.hidden.task_id)
            if self.final_breakdown.recovered_turn is not None:
                self.hidden.recovered_turn = self.final_breakdown.recovered_turn
            obs = self._build_observation(session_feedback_signal=float(aux["satisfaction_proxy"]))
            info["final_grade"] = self.final_breakdown.model_dump()

        return StepResult(
            observation=obs,
            reward=float(round(reward_value, 6)),
            done=done,
            info=info,
        )

    def current_grade(self) -> FinalGradeBreakdown:
        if self.hidden is None:
            raise RuntimeError("Environment not initialized.")
        return final_grade(self.trajectory, self.hidden.task_id)