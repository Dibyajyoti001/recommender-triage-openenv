from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .candidate_pool import build_candidate_pool
from .data import DEFAULT_SEEDS
from .graders import FinalGradeBreakdown, TrajectoryStep, final_grade
from .models import (
    Action,
    EnvironmentState,
    HiddenState,
    MemorySummary,
    Observation,
    RecentInteraction,
    RewardBreakdown,
    StepResult,
)
from .reward import (
    category_fatigue_update,
    compute_step_reward,
    confidence_bucket,
    feedback_bucket,
    pressure_bucket,
    repetition_pressure,
    update_patience,
)
from .tasks import CATEGORY_NAMES, GLOBAL, K, TaskConfig, TaskSpec, get_task_config


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v, 1e-8)
    s = float(v.sum())
    return v / s if s > 0 else np.ones_like(v) / len(v)


class RecommenderEnv:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.base_seed = seed if seed is not None else 0
        self.rng = np.random.default_rng(self.base_seed)

        self.task_cfg: Optional[TaskConfig] = None
        self.task_spec: Optional[TaskSpec] = None
        self.hidden: Optional[HiddenState] = None

        self.latest_reward_breakdown: Optional[RewardBreakdown] = None
        self.candidate_items = []
        self.recent_interactions: List[RecentInteraction] = []
        self.trajectory: List[TrajectoryStep] = []
        self.final_breakdown: Optional[FinalGradeBreakdown] = None

        self._zero_patience_streak = 0

    def reset(self, task_id: str = "task_1", seed: Optional[int] = None) -> Observation:
        self.task_cfg = get_task_config(task_id)
        self.task_spec = self.task_cfg.to_spec()

        effective_seed = seed if seed is not None else DEFAULT_SEEDS.get(task_id, 0) + self.base_seed
        self.rng = np.random.default_rng(effective_seed)

        m = np.asarray(self.task_cfg.memory_pref, dtype=float)
        z = np.asarray(self.task_cfg.session_intent, dtype=float)

        # Observation noise is small and task-dependent, but state remains inside the task envelope.
        eps = self.task_cfg.epsilon_obs
        if eps > 0:
            m = _normalize(m + self.rng.uniform(0.0, eps * 0.05, size=K))
            z = _normalize(z + self.rng.uniform(0.0, eps * 0.05, size=K))

        # Patience is task-generated, not a fixed global constant.
        # Keep it high but not identical across tasks/episodes.
        p_init = float(np.clip(self.rng.uniform(0.78, 0.92), 0.0, 1.0))

        self.hidden = HiddenState(
            task_id=task_id,
            turn=0,
            m=[float(x) for x in m],
            z=[float(x) for x in z],
            F_cat=[0.0 for _ in range(K)],
            F_item={},
            nu=float(self.task_cfg.nu),
            p=p_init,
            chi=float(self.task_cfg.chi_init),
            history_item_ids=[],
            history_category_ids=[],
            drift_turn=None,
            recovered_turn=None,
            last_feedback_bucket="neutral",
            rng_seed=effective_seed,
        )

        self.recent_interactions = []
        self.trajectory = []
        self.final_breakdown = None
        self.latest_reward_breakdown = None
        self._zero_patience_streak = 0

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
        )
        return self._build_observation(session_feedback_signal=0.5)

    def state(self) -> EnvironmentState:
        if self.hidden is None or self.task_spec is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return EnvironmentState(
            hidden_state=self.hidden,
            task_spec=self.task_spec,
            trajectory_length=len(self.trajectory),
            latest_reward_breakdown=self.latest_reward_breakdown,
        )

    def _memory_summary(self) -> MemorySummary:
        assert self.hidden is not None
        top_idxs = sorted(range(K), key=lambda i: self.hidden.m[i], reverse=True)[:3]
        return MemorySummary(
            top_categories=top_idxs,
            top_category_names=[CATEGORY_NAMES[i] for i in top_idxs],  # type: ignore[list-item]
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

        if self.hidden.history_category_ids:
            last_cat = self.hidden.history_category_ids[-1]
            rho = repetition_pressure(
                self.hidden.history_category_ids[:-1],
                last_cat,
                GLOBAL.repetition_window,
            )
        else:
            rho = 0.0

        return Observation(
            task_id=self.hidden.task_id,
            task_name=get_task_config(self.hidden.task_id).task_name,
            turn_id=self.hidden.turn,
            max_turns=GLOBAL.T_max,
            memory_summary=self._memory_summary(),
            recent_interactions=self.recent_interactions[-5:],
            candidate_items=self.candidate_items,
            repetition_counts=repetition_counts,
            repetition_pressure_bucket=pressure_bucket(rho),  # type: ignore[arg-type]
            memory_confidence_bucket=confidence_bucket(self.hidden.chi),  # type: ignore[arg-type]
            memory_confidence=float(round(self.hidden.chi, 6)),
            session_feedback_signal=float(round(session_feedback_signal, 6)),
            done_hint="Session continues until max turns or patience collapses repeatedly.",
        )

    def _update_memory_confidence(
        self,
        chosen_category_id: int,
        pre_m: List[float],
        pre_z: List[float],
        prev_chi: float,
    ) -> float:
        """
        More faithful contradiction/support update:
        - contradiction if live top beats memory top clearly and choice follows live not memory
        - support if memory top remains competitive and choice follows memory
        """
        k_m = int(np.argmax(np.asarray(pre_m)))
        k_z = int(np.argmax(np.asarray(pre_z)))

        memory_advantage = pre_m[k_m] - pre_z[k_z]
        live_advantage = pre_z[k_z] - pre_m[k_m]

        contradicts_memory = (live_advantage > 0.05 and chosen_category_id == k_z and k_z != k_m)
        supports_memory = (memory_advantage >= -0.05 and chosen_category_id == k_m)

        chi = prev_chi
        if contradicts_memory:
            chi -= GLOBAL.eta_chi
        if supports_memory:
            chi += GLOBAL.beta_chi

        return float(np.clip(chi, 0.0, 1.0))

    def _drift_targets(self, current_z: np.ndarray, current_m: np.ndarray, task_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        phi_t: continuation target
        psi_t: drift target
        Keep task-3 conflict strong but not a single rigid hand-scripted case.
        """
        phi_t = _normalize(0.80 * current_z + 0.20 * current_m)

        if task_id == "task_3":
            # Shift away from memory-dominant category toward a competing live category.
            k_m = int(np.argmax(current_m))
            k_z = int(np.argmax(current_z))
            if k_z == k_m:
                # If they accidentally align, push conflict toward a neighboring category.
                alt = (k_m + 1) % K
            else:
                alt = k_z

            psi_t = np.full(K, 0.08, dtype=float)
            psi_t[alt] = 0.56
            psi_t[k_m] = 0.12
            psi_t = _normalize(psi_t)
        elif task_id == "task_2":
            dominant = int(np.argmax(current_z))
            psi_t = np.full(K, 0.10, dtype=float)
            psi_t[dominant] = 0.30
            psi_t[(dominant + 1) % K] = 0.25
            psi_t[(dominant + 2) % K] = 0.25
            psi_t = _normalize(psi_t)
        else:
            psi_t = _normalize(0.60 * current_m + 0.40 * current_z)

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
        pre_chi = float(self.hidden.chi)
        pre_p = float(self.hidden.p)
        pre_argmax_z = int(np.argmax(np.asarray(pre_z)))

        cat_fatigue_val = float(self.hidden.F_cat[chosen.category_id])
        item_fatigue_val = float(self.hidden.F_item.get(chosen.item_id, 0.0))

        reward_value, reward_breakdown, aux = compute_step_reward(
            alpha=self.task_cfg.alpha,
            eta_q=GLOBAL.eta_q,
            z=pre_z,
            m=pre_m,
            item=chosen,
            category_fatigue_value=cat_fatigue_val,
            item_fatigue_value=item_fatigue_val,
            history_categories=self.hidden.history_category_ids,
            nu=self.hidden.nu,
            chi_t=pre_chi,
            exploration_flag=action.exploration_flag,
            confidence_score=action.confidence_score,
            repetition_window=GLOBAL.repetition_window,
            w_c=GLOBAL.w_c,
            w_i=GLOBAL.w_i,
        )
        self.latest_reward_breakdown = reward_breakdown

        new_F_cat = list(self.hidden.F_cat)
        for k in range(K):
            new_F_cat[k] = category_fatigue_update(
                current=new_F_cat[k],
                lambda_c=self.task_cfg.lambda_c,
                delta_c=self.task_cfg.delta_c,
                chosen=(k == chosen.category_id),
            )

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

        dissatisfaction_value = (
            GLOBAL.gamma1 * (1.0 - aux["relevance"])
            + GLOBAL.gamma2 * aux["fatigue_cost"]
            + GLOBAL.gamma3 * aux["novelty_violation"]
        )
        new_p = update_patience(pre_p, aux["relevance"], dissatisfaction_value)

        z_arr = np.asarray(pre_z, dtype=float)
        m_arr = np.asarray(pre_m, dtype=float)
        phi_t, psi_t = self._drift_targets(z_arr, m_arr, self.hidden.task_id)

        if new_p > self.task_cfg.tau:
            new_z = (1.0 - self.task_cfg.mu) * z_arr + self.task_cfg.mu * phi_t
        else:
            new_z = (1.0 - self.task_cfg.kappa) * z_arr + self.task_cfg.kappa * psi_t
        new_z = _normalize(new_z)

        post_argmax_z = int(np.argmax(new_z))
        drift_happened = (new_p <= self.task_cfg.tau) and (post_argmax_z != pre_argmax_z)

        if drift_happened and self.hidden.drift_turn is None:
            self.hidden.drift_turn = self.hidden.turn

        new_chi = self._update_memory_confidence(
            chosen_category_id=chosen.category_id,
            pre_m=pre_m,
            pre_z=pre_z,
            prev_chi=pre_chi,
        )

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

        fb_bucket = feedback_bucket(aux["satisfaction_proxy"])
        self.recent_interactions.append(
            RecentInteraction(
                turn_id=self.hidden.turn,
                item_id=chosen.item_id,
                category_id=chosen.category_id,
                category_name=chosen.category_name,  # type: ignore[arg-type]
                confidence_score=action.confidence_score,
                exploration_flag=action.exploration_flag,
                reward=reward_value,
                satisfaction_proxy=float(aux["satisfaction_proxy"]),
                feedback_bucket=fb_bucket,  # type: ignore[arg-type]
            )
        )

        self.hidden.turn += 1
        self.hidden.m = list(pre_m)
        self.hidden.z = [float(x) for x in new_z]
        self.hidden.F_cat = [float(x) for x in new_F_cat]
        self.hidden.F_item = {int(k): float(v) for k, v in new_F_item.items()}
        self.hidden.p = float(new_p)
        self.hidden.chi = float(new_chi)
        self.hidden.last_feedback_bucket = fb_bucket  # type: ignore[assignment]

        # Exact early termination rule: p_t = 0 for L consecutive turns
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
    