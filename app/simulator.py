from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .candidate_pool import build_candidate_pool
from .models import (
    Action,
    CandidateItem,
    CounterfactualAuditResult,
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
    engagement_bucket,
    confidence_bucket,
    feedback_bucket,
    pressure_bucket,
    update_patience,
)
from .tasks import CATEGORY_NAMES, GLOBAL, K, TaskConfig, _light_user_heterogeneity, get_task_config
from .graders import final_grade


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.maximum(arr, 0.0)
    s = float(arr.sum())
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _rolling_mean(values: List[float], window: int) -> Tuple[float, int]:
    if window <= 0 or not values:
        return 0.0, 0
    clipped = values[-window:]
    return float(sum(clipped) / len(clipped)), len(clipped)


REGIME_NAMES: Tuple[str, ...] = ("stable", "drifting", "fatigued", "distressed")


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
    trust_before: float
    trust_after: float
    observed_feedback: float
    feedback_volatility: float
    repetition_pressure: float
    novelty_violation: float
    resource_pressure: float
    risk_exposure: float
    diversity_pressure: float
    platform_gain: float
    budget_remaining: float
    risk_tolerance: float
    latency_budget: float
    calibration_target: float
    slot_type: str
    saturated: bool
    trust_collapsed: bool
    risk_collapsed: bool
    diversity_collapsed: bool
    regime: int
    latent_vol: float
    exploration_flag: bool
    confidence_score: float


@dataclass
class CounterfactualAuditSnapshot:
    turn_id: int
    fragility: float
    chosen_action: Action
    chosen_slot_type: str
    hidden_state: HiddenState
    candidate_items: List[CandidateItem]
    recent_interactions: List[RecentInteraction]
    zero_patience_streak: int
    replay_seeds: List[int]


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
        self.audit_snapshots: List[CounterfactualAuditSnapshot] = []
        self.audit_results: List[CounterfactualAuditResult] = []
        self.audit_enabled = True
        self._last_audit_turn: Optional[int] = None
        self._zero_patience_streak = 0

    def _feedback_volatility(
        self,
        history: Optional[List[float]] = None,
        floor_override: Optional[float] = None,
        latent_override: Optional[float] = None,
    ) -> float:
        assert self.task_cfg is not None
        values = self.hidden.feedback_history if history is None else history
        window = values[-self.task_cfg.volatility_window :]
        if len(window) < 2:
            base = 0.0
        else:
            base = float(np.var(np.asarray(window, dtype=float)))
        if floor_override is not None:
            floor = float(floor_override)
        else:
            floor = float(self.hidden.volatility_floor) if self.hidden is not None else 0.0
        if latent_override is not None:
            latent_shadow = 0.12 * float(latent_override)
        elif self.hidden is not None:
            latent_shadow = 0.12 * float(self.hidden.latent_vol)
        else:
            latent_shadow = 0.0
        return max(base, floor, latent_shadow)

    def _recent_concentration(self, chosen_category: int) -> float:
        assert self.task_cfg is not None
        recent = list(self.hidden.history_category_ids[-(GLOBAL.repetition_window - 1):]) + [chosen_category]
        if not recent:
            return 0.0
        counts = {cat: recent.count(cat) for cat in set(recent)}
        return float(max(counts.values()) / len(recent))

    def _realized_item(self, chosen):
        assert self.task_cfg is not None
        is_dud = False
        realized = chosen

        if (
            self.task_cfg.dud_probability > 0.0
            and chosen.slot_type == "live_best_fresh"
            and self.rng.random() < self.task_cfg.dud_probability
        ):
            lo, hi = self.task_cfg.dud_quality_range
            realized_quality = float(self.rng.uniform(lo, hi))
            realized = chosen.model_copy(update={"quality": realized_quality})
            is_dud = True

        return realized, is_dud

    def _update_trust(
        self,
        *,
        pre_trust: float,
        y_true: float,
        confidence_score: float,
        calibration_target: float,
        feedback_volatility: float,
        saturation_triggered: bool,
        resource_pressure: float,
        regime: int,
        latent_vol: float,
    ) -> float:
        assert self.task_cfg is not None

        cal_gap = abs(confidence_score - calibration_target)
        underclaim = max(0.0, y_true - confidence_score)

        gain = self.task_cfg.trust_gain * y_true * max(0.0, 1.0 - cal_gap)
        loss = self.task_cfg.trust_loss * (1.0 - y_true) * (0.5 + 0.5 * confidence_score)
        loss += self.task_cfg.trust_underclaim * underclaim
        loss += self.task_cfg.trust_volatility * feedback_volatility
        loss += self.task_cfg.resource_pressure_loss * resource_pressure

        regime_loss_mult = self.task_cfg.regime_trust_loss_multiplier[regime]
        gain *= max(0.80, 1.0 - 0.10 * latent_vol * regime)
        loss *= regime_loss_mult
        if regime == 2:
            loss += 0.05 * latent_vol
        elif regime == 3:
            loss += 0.12 * latent_vol

        if saturation_triggered:
            loss += 0.10

        return float(np.clip(pre_trust + gain - loss, 0.0, 1.0))

    def _update_regime(
        self,
        *,
        trust: float,
        patience: float,
        repetition_pressure: float,
        latent_vol: float,
        alignment: float,
    ) -> int:
        assert self.hidden is not None
        assert self.task_cfg is not None

        current = int(self.hidden.regime)
        probs = np.full(4, 1e-4, dtype=float)
        probs[current] += float(self.task_cfg.regime_sticky)

        stable_relief = float(
            np.clip(
                0.40 * np.clip((trust - 0.60) / 0.25, 0.0, 1.0)
                + 0.30 * np.clip((patience - 0.60) / 0.25, 0.0, 1.0)
                + 0.15 * np.clip((0.45 - repetition_pressure) / 0.45, 0.0, 1.0)
                + 0.15 * np.clip((0.30 - latent_vol) / 0.30, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        drift_pressure = float(
            np.clip(
                0.60 * np.clip((0.35 - alignment) / 0.35, 0.0, 1.0)
                + 0.40 * np.clip((latent_vol - 0.35) / 0.40, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        fatigue_pressure = float(np.clip((repetition_pressure - 0.52) / 0.30, 0.0, 1.0))
        distress_pressure = float(
            np.clip(
                max(
                    np.clip((0.46 - trust) / 0.30, 0.0, 1.0),
                    np.clip((0.36 - patience) / 0.28, 0.0, 1.0),
                )
                + 0.25 * fatigue_pressure
                + 0.20 * np.clip((latent_vol - 0.55) / 0.30, 0.0, 1.0),
                0.0,
                1.0,
            )
        )

        probs[0] += 0.20 + 0.40 * stable_relief
        probs[1] += 0.12 + 0.55 * drift_pressure
        probs[2] += 0.10 + 0.60 * fatigue_pressure
        probs[3] += 0.08 + 0.70 * distress_pressure

        if current == 0:
            probs *= np.asarray([1.00, 1.05, 1.10, 1.15])
        elif current == 1:
            probs *= np.asarray([0.80, 1.00, 1.05, 1.10])
        elif current == 2:
            probs *= np.asarray([0.65, 0.85, 1.00, 1.12])
        else:
            probs *= np.asarray([0.50, 0.70, 0.88, 1.00])

        probs = probs / float(probs.sum())
        return int(self.rng.choice(4, p=probs))

    def _update_latent_vol(self, prev_latent_vol: float, regime: int) -> float:
        assert self.task_cfg is not None
        regime_targets = [0.08, 0.26, 0.44, 0.68]
        target = regime_targets[int(max(0, min(3, regime)))]
        noise_scale = 0.015 + 0.035 * self.task_cfg.regime_noise_multiplier[regime]
        updated = (
            self.task_cfg.latent_vol_revert * prev_latent_vol
            + (1.0 - self.task_cfg.latent_vol_revert) * target
            + noise_scale * float(self.rng.normal())
        )
        return float(np.clip(updated, 0.0, 1.0))

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        self.task_cfg = get_task_config(task_id)
        self.task_cfg = _light_user_heterogeneity(self.task_cfg, self.rng)
        self.recent_interactions = []
        self.trajectory = []
        self.latest_reward_breakdown = None
        self.final_breakdown = None
        self.audit_snapshots = []
        self.audit_results = []
        self.audit_enabled = True
        self._last_audit_turn = None
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
            trust=float(self.task_cfg.trust_init),
            budget_remaining=float(self.task_cfg.budget_init),
            risk_tolerance=float(self.task_cfg.risk_tolerance_init),
            latency_budget=float(self.task_cfg.latency_budget_init),
            feedback_history=[],
            saturation_turn=None,
            trust_collapsed=False,
            risk_collapsed=False,
            diversity_collapsed=False,
            risk_noise_floor=0.0,
            volatility_floor=0.0,
            risk_history=[],
            diversity_history=[],
            rng_seed=self.seed,
            regime=int(self.task_cfg.regime_init),
            latent_vol=float(self.task_cfg.latent_vol_init),
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
            trust_level=self.hidden.trust,
            budget_remaining=self.hidden.budget_remaining,
            risk_tolerance=self.hidden.risk_tolerance,
            latency_budget=self.hidden.latency_budget,
            diversity_collapsed=self.hidden.diversity_collapsed,
            regime=self.hidden.regime,
            latent_vol=self.hidden.latent_vol,
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
            audit_results=self.audit_results,
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
        volatility = self._feedback_volatility()

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
            feedback_volatility=float(round(volatility, 6)),
            trust_signal=float(round(self.hidden.trust, 6)),
            engagement_signal=float(round(self.hidden.p, 6)),
            budget_remaining=float(round(self.hidden.budget_remaining, 6)),
            risk_tolerance=float(round(self.hidden.risk_tolerance, 6)),
            latency_budget=float(round(self.hidden.latency_budget, 6)),
            engagement_bucket=engagement_bucket(self.hidden.p),
            done_hint="Session continues until max turns or engagement collapses repeatedly.",
        )

    def _counterfactual_fragility(
        self,
        *,
        feedback_volatility: float,
        trust_level: float,
        budget_remaining: float,
        risk_tolerance: float,
        H_topic: List[float],
        z: List[float],
    ) -> float:
        volatility_pressure = min(1.0, max(0.0, feedback_volatility) / 0.08)
        repetition_pressure = _dot(H_topic, z)
        trust_pressure = min(1.0, max(0.0, 0.50 - trust_level) / 0.50)
        risk_pressure = min(1.0, max(0.0, 0.40 - risk_tolerance) / 0.40)
        budget_pressure = min(1.0, max(0.0, 0.40 - budget_remaining) / 0.40)
        return float(
            np.clip(
                0.30 * volatility_pressure
                + 0.25 * repetition_pressure
                + 0.20 * trust_pressure
                + 0.15 * risk_pressure
                + 0.10 * budget_pressure,
                0.0,
                1.0,
            )
        )

    def _future_replay_seeds(self, horizon: int) -> List[int]:
        temp_rng = np.random.default_rng()
        temp_rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        return [
            int(temp_rng.integers(0, np.iinfo(np.uint32).max))
            for _ in range(max(1, horizon))
        ]

    def _candidate_repetition_hint(self, item: CandidateItem) -> float:
        hint = item.metadata.get("repetition_hint", 0.0)
        return float(np.clip(float(hint), 0.0, 1.0))

    def _select_safe_alternative(
        self,
        candidates: List[CandidateItem],
        *,
        chosen_item_id: int,
    ) -> CandidateItem:
        pool = [item for item in candidates if item.item_id != chosen_item_id] or list(candidates)
        return max(
            pool,
            key=lambda item: (
                0.40 * item.quality
                + 0.15 * item.engagement
                + (0.03 if item.freshness == "fresh" else 0.0)
                - 0.20 * item.risk
                - 0.15 * item.cost
                - 0.10 * item.latency
                - 0.15 * self._candidate_repetition_hint(item)
            ),
        )

    def _select_risky_alternative(
        self,
        candidates: List[CandidateItem],
        *,
        chosen_item_id: int,
        safe_item_id: int,
    ) -> CandidateItem:
        pool = [
            item
            for item in candidates
            if item.item_id not in {chosen_item_id, safe_item_id}
        ] or [item for item in candidates if item.item_id != safe_item_id] or list(candidates)
        return max(
            pool,
            key=lambda item: (
                0.62 * item.quality
                + 0.18 * item.engagement
                + (0.08 if item.freshness == "fresh" else 0.02 if item.freshness == "novel" else -0.02)
                + (0.06 if item.slot_type in {"live_best_fresh", "fatigue_trap"} else 0.0)
                - 0.04 * item.cost
                - 0.03 * item.risk
                - 0.02 * item.latency
            ),
        )

    def _branch_action_for_item(
        self,
        item: CandidateItem,
        *,
        trust_signal: float,
        feedback_volatility: float,
        repetition_pressure: float,
    ) -> Action:
        explore = bool(
            item.slot_type in {"exploration_option", "conflict_option"}
            or item.freshness == "novel"
            or repetition_pressure >= 0.67
            or trust_signal < 0.50
        )
        confidence = 0.88 - 2.0 * feedback_volatility
        if explore:
            confidence -= 0.12
        if trust_signal < 0.45:
            confidence -= 0.08
        confidence = float(np.clip(confidence, 0.20, 0.92))
        return Action(
            recommended_item_id=item.item_id,
            exploration_flag=explore,
            confidence_score=confidence,
        )

    def _counterfactual_followup_action(self, obs: Observation) -> Action:
        candidates = obs.candidate_items
        if not candidates:
            return Action(recommended_item_id=0, exploration_flag=False, confidence_score=0.0)

        top_memory_cat = obs.memory_summary.top_categories[0] if obs.memory_summary.top_categories else 0
        recent_majority: Optional[int] = None
        if obs.recent_interactions:
            recent_cats = [item.category_id for item in obs.recent_interactions[-3:]]
            recent_majority = max(set(recent_cats), key=recent_cats.count)

        best_item = candidates[0]
        best_score = float("-inf")

        for item in candidates:
            repetition_pen = obs.repetition_counts[item.category_id] * 0.08
            memory_bonus = 0.12 if item.category_id == top_memory_cat else 0.0
            freshness_bonus = 0.06 if item.freshness == "fresh" else (0.08 if item.freshness == "novel" else -0.02)
            anti_repeat_bonus = 0.05 if recent_majority is not None and item.category_id != recent_majority else 0.0
            trust_repair_bonus = 0.05 if obs.trust_signal < 0.55 and item.slot_type in {"balanced_bridge", "exploration_option"} else 0.0
            volatility_pen = 0.08 if obs.feedback_volatility > 0.05 and item.slot_type == "live_best_fresh" else 0.0
            budget_pen = 0.12 * item.cost / max(obs.budget_remaining, 0.10)
            risk_pen = 0.14 * item.risk / max(obs.risk_tolerance, 0.10)
            latency_pen = 0.10 * item.latency / max(obs.latency_budget, 0.10)

            score = (
                0.55 * item.quality
                + 0.20 * item.engagement
                + memory_bonus
                + freshness_bonus
                + anti_repeat_bonus
                + trust_repair_bonus
                - repetition_pen
                - volatility_pen
                - budget_pen
                - risk_pen
                - latency_pen
            )

            if score > best_score:
                best_score = score
                best_item = item

        return self._branch_action_for_item(
            best_item,
            trust_signal=obs.trust_signal,
            feedback_volatility=obs.feedback_volatility,
            repetition_pressure=_dot(self.hidden.H_topic, self.hidden.z) if self.hidden is not None else 0.0,
        )

    def _make_replay_env(self, snapshot: CounterfactualAuditSnapshot) -> "RecommendationPolicyEnvironment":
        replay_env = RecommendationPolicyEnvironment(seed=snapshot.hidden_state.rng_seed)
        replay_env.rng = np.random.default_rng(snapshot.replay_seeds[0])
        replay_env.task_cfg = get_task_config(snapshot.hidden_state.task_id)
        replay_env.hidden = snapshot.hidden_state.model_copy(deep=True)
        replay_env.candidate_items = [item.model_copy(deep=True) for item in snapshot.candidate_items]
        replay_env.recent_interactions = [item.model_copy(deep=True) for item in snapshot.recent_interactions]
        replay_env.trajectory = []
        replay_env.latest_reward_breakdown = None
        replay_env.final_breakdown = None
        replay_env.audit_snapshots = []
        replay_env.audit_results = []
        replay_env.audit_enabled = False
        replay_env._last_audit_turn = None
        replay_env._zero_patience_streak = snapshot.zero_patience_streak
        return replay_env

    def _branch_value(
        self,
        replay_env: "RecommendationPolicyEnvironment",
        rewards: List[float],
    ) -> float:
        if not rewards:
            return 0.0

        return_score = float(np.clip(0.5 + 0.5 * (sum(rewards) / len(rewards)), 0.0, 1.0))
        terminal_trust = replay_env.hidden.trust if replay_env.hidden is not None else 0.0
        risk_safe = 1.0
        if replay_env.trajectory:
            risk_safe = float(
                np.clip(
                    1.0 - sum(step.risk_exposure for step in replay_env.trajectory) / len(replay_env.trajectory),
                    0.0,
                    1.0,
                )
            )
        collapse_free = 1.0
        if replay_env.hidden is not None and (
            replay_env.hidden.trust_collapsed
            or replay_env.hidden.risk_collapsed
            or replay_env.hidden.diversity_collapsed
        ):
            collapse_free = 0.0
        regime_health = 1.0
        if replay_env.trajectory:
            regime_health = float(
                np.clip(
                    1.0
                    - sum(
                        (step.regime / 3.0) * (0.40 + 0.60 * step.latent_vol)
                        for step in replay_env.trajectory
                    )
                    / len(replay_env.trajectory),
                    0.0,
                    1.0,
                )
            )

        return float(
            np.clip(
                0.50 * return_score
                + 0.20 * terminal_trust
                + 0.15 * risk_safe
                + 0.10 * collapse_free
                + 0.05 * regime_health,
                0.0,
                1.0,
            )
        )

    def _run_counterfactual_branch(
        self,
        snapshot: CounterfactualAuditSnapshot,
        first_action: Action,
    ) -> float:
        replay_env = self._make_replay_env(snapshot)
        rewards: List[float] = []
        obs = replay_env._build_observation(session_feedback_signal=0.5)

        for idx, seed in enumerate(snapshot.replay_seeds):
            replay_env.rng = np.random.default_rng(seed)
            action = first_action if idx == 0 else replay_env._counterfactual_followup_action(obs)
            result = replay_env.step(action)
            rewards.append(result.reward)
            obs = result.observation
            if result.done:
                break

        return self._branch_value(replay_env, rewards)

    def _evaluate_counterfactual_audit(
        self,
        snapshot: CounterfactualAuditSnapshot,
    ) -> CounterfactualAuditResult:
        chosen_item = next(
            (item for item in snapshot.candidate_items if item.item_id == snapshot.chosen_action.recommended_item_id),
            snapshot.candidate_items[0],
        )
        safe_item = self._select_safe_alternative(
            snapshot.candidate_items,
            chosen_item_id=chosen_item.item_id,
        )
        risky_item = self._select_risky_alternative(
            snapshot.candidate_items,
            chosen_item_id=chosen_item.item_id,
            safe_item_id=safe_item.item_id,
        )

        repetition_pressure = _dot(snapshot.hidden_state.H_topic, snapshot.hidden_state.z)
        safe_action = self._branch_action_for_item(
            safe_item,
            trust_signal=snapshot.hidden_state.trust,
            feedback_volatility=self._feedback_volatility(
                snapshot.hidden_state.feedback_history,
                floor_override=snapshot.hidden_state.volatility_floor,
                latent_override=snapshot.hidden_state.latent_vol,
            ),
            repetition_pressure=repetition_pressure,
        )
        risky_action = self._branch_action_for_item(
            risky_item,
            trust_signal=snapshot.hidden_state.trust,
            feedback_volatility=self._feedback_volatility(
                snapshot.hidden_state.feedback_history,
                floor_override=snapshot.hidden_state.volatility_floor,
                latent_override=snapshot.hidden_state.latent_vol,
            ),
            repetition_pressure=repetition_pressure,
        )

        chosen_value = self._run_counterfactual_branch(snapshot, snapshot.chosen_action)
        safe_value = self._run_counterfactual_branch(snapshot, safe_action)
        risky_value = self._run_counterfactual_branch(snapshot, risky_action)

        scale = max(
            0.05,
            float(np.std([chosen_value, safe_value, risky_value]) * 2.0),
        )
        safe_component = 1.0 if chosen_item.item_id == safe_item.item_id else float(
            np.clip(0.5 + (chosen_value - safe_value) / scale, 0.0, 1.0)
        )
        risky_component = float(np.clip(0.5 + (chosen_value - risky_value) / scale, 0.0, 1.0))
        audit_score = float(np.clip(0.70 * safe_component + 0.30 * risky_component, 0.0, 1.0))

        return CounterfactualAuditResult(
            turn_id=snapshot.turn_id,
            fragility=float(round(snapshot.fragility, 6)),
            chosen_value=float(round(chosen_value, 6)),
            safe_value=float(round(safe_value, 6)),
            risky_value=float(round(risky_value, 6)),
            safe_advantage=float(round(chosen_value - safe_value, 6)),
            risky_advantage=float(round(chosen_value - risky_value, 6)),
            audit_score=float(round(audit_score, 6)),
            chosen_slot_type=chosen_item.slot_type,
            safe_slot_type=safe_item.slot_type,
            risky_slot_type=risky_item.slot_type,
        )

    def _maybe_record_counterfactual_snapshot(
        self,
        *,
        chosen: CandidateItem,
        action: Action,
        feedback_volatility: float,
        trust_level: float,
        budget_remaining: float,
        risk_tolerance: float,
        H_topic: List[float],
        z: List[float],
    ) -> None:
        assert self.hidden is not None
        assert self.task_cfg is not None

        if not self.audit_enabled or self.hidden.task_id not in {"task_4", "task_5"}:
            return
        if self.task_cfg.counterfactual_audit_budget <= 0:
            return
        if len(self.audit_snapshots) >= self.task_cfg.counterfactual_audit_budget:
            return
        if self.hidden.turn >= GLOBAL.T_max - 2:
            return
        if self._last_audit_turn is not None and self.hidden.turn - self._last_audit_turn < 3:
            return

        fragility = self._counterfactual_fragility(
            feedback_volatility=feedback_volatility,
            trust_level=trust_level,
            budget_remaining=budget_remaining,
            risk_tolerance=risk_tolerance,
            H_topic=H_topic,
            z=z,
        )
        if fragility < self.task_cfg.counterfactual_fragility_threshold:
            return

        replay_seeds = self._future_replay_seeds(self.task_cfg.counterfactual_audit_horizon)
        self.audit_snapshots.append(
            CounterfactualAuditSnapshot(
                turn_id=self.hidden.turn,
                fragility=fragility,
                chosen_action=action.model_copy(deep=True),
                chosen_slot_type=chosen.slot_type,
                hidden_state=self.hidden.model_copy(deep=True),
                candidate_items=[item.model_copy(deep=True) for item in self.candidate_items],
                recent_interactions=[item.model_copy(deep=True) for item in self.recent_interactions],
                zero_patience_streak=self._zero_patience_streak,
                replay_seeds=replay_seeds,
            )
        )
        self._last_audit_turn = self.hidden.turn

    def _ensure_audit_results(self) -> None:
        if not self.audit_enabled or self.audit_results or not self.audit_snapshots:
            return
        self.audit_results = [
            self._evaluate_counterfactual_audit(snapshot)
            for snapshot in self.audit_snapshots
        ]

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
        pre_trust = float(self.hidden.trust)
        pre_budget_remaining = float(self.hidden.budget_remaining)
        pre_risk_tolerance = float(self.hidden.risk_tolerance)
        pre_latency_budget = float(self.hidden.latency_budget)
        pre_regime = int(self.hidden.regime)
        pre_latent_vol = float(self.hidden.latent_vol)
        pre_volatility = self._feedback_volatility()
        pre_argmax_z = int(np.argmax(np.asarray(pre_z)))
        pre_alignment = _dot(pre_z, pre_m)
        pre_rep_pressure = _dot(pre_H, pre_z)

        chosen_x = list(
            chosen.topic_vector
            or chosen.metadata.get("topic_vector")
            or [1.0 if i == chosen.category_id else 0.0 for i in range(K)]
        )
        realized_item, is_dud = self._realized_item(chosen)

        drift_trigger_turn = 4 + (self.hidden.rng_seed % 4)
        new_regime = self._update_regime(
            trust=pre_trust,
            patience=pre_p,
            repetition_pressure=pre_rep_pressure,
            latent_vol=pre_latent_vol,
            alignment=pre_alignment,
        )
        if (
            self.hidden.task_id == "task_3"
            and self.hidden.turn == drift_trigger_turn
            and self.hidden.drift_turn is None
        ):
            new_regime = max(new_regime, 1)
        new_latent_vol = self._update_latent_vol(pre_latent_vol, new_regime)
        if new_regime == 3:
            new_latent_vol = max(new_latent_vol, 0.52)
        elif new_regime == 2:
            new_latent_vol = max(new_latent_vol, 0.34)
        elif new_regime == 1:
            new_latent_vol = max(new_latent_vol, 0.18)

        self._maybe_record_counterfactual_snapshot(
            chosen=chosen,
            action=action,
            feedback_volatility=pre_volatility,
            trust_level=pre_trust,
            budget_remaining=pre_budget_remaining,
            risk_tolerance=pre_risk_tolerance,
            H_topic=pre_H,
            z=pre_z,
        )

        category_fatigue_val = float(self.hidden.F_cat[chosen.category_id])
        item_fatigue_val = float(self.hidden.F_item.get(chosen.item_id, 0.0))

        reward_value, reward_breakdown, aux = compute_step_reward(
            alpha=self.task_cfg.alpha,
            eta_q=GLOBAL.eta_q,
            z=pre_z,
            m=pre_m,
            item=realized_item,
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
            trust_level=pre_trust,
            trust_sensitivity=self.task_cfg.trust_sensitivity,
            feedback_volatility=pre_volatility,
            calibration_decay_k=self.task_cfg.calibration_decay_k,
            confidence_penalty_weight=self.task_cfg.confidence_reward_weight,
            budget_remaining=pre_budget_remaining,
            risk_tolerance=pre_risk_tolerance,
            latency_budget=pre_latency_budget,
            concentration_recent=self._recent_concentration(chosen.category_id),
            satisfaction_cap=(
                self.task_cfg.collapsed_satisfaction_cap
                if self.hidden.trust_collapsed
                else None
            ),
        )
        self.latest_reward_breakdown = reward_breakdown

        fatigue_mult = self.task_cfg.regime_fatigue_multiplier[new_regime] * (1.0 + 0.35 * new_latent_vol)
        new_F_topic = [
            float(self.task_cfg.lambda_F * pre_F[i] + self.task_cfg.delta_F * fatigue_mult * chosen_x[i])
            for i in range(K)
        ]

        h_lambda = self.task_cfg.lambda_H
        if self.hidden.saturation_turn is not None and self.task_cfg.lambda_H_recovery > 0.0:
            h_lambda = self.task_cfg.lambda_H_recovery

        new_H_topic = [
            float(h_lambda * pre_H[i] + (1.0 - h_lambda) * chosen_x[i])
            for i in range(K)
        ]
        new_H_topic = list(_normalize(np.asarray(new_H_topic, dtype=float)))
        saturated_now = max(new_H_topic) >= self.task_cfg.saturation_threshold
        saturation_triggered = (
            self.task_cfg.saturation_threshold < 1.0
            and self.hidden.saturation_turn is None
            and saturated_now
        )
        if saturation_triggered:
            self.hidden.saturation_turn = self.hidden.turn

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
        observed_feedback = y_t
        effective_noise_std = (
            self.task_cfg.feedback_noise_std
            + 0.04 * new_latent_vol * self.task_cfg.regime_noise_multiplier[new_regime]
            + float(self.hidden.risk_noise_floor)
        )
        if effective_noise_std > 0.0:
            observed_feedback = float(
                np.clip(
                    y_t + self.rng.normal(0.0, effective_noise_std),
                    0.0,
                    1.0,
                )
            )

        new_feedback_history = list(self.hidden.feedback_history)
        new_feedback_history.append(observed_feedback)
        post_volatility = self._feedback_volatility(
            new_feedback_history,
            latent_override=new_latent_vol,
        )

        new_budget_remaining = float(
            np.clip(
                pre_budget_remaining + self.task_cfg.budget_recovery - self.task_cfg.budget_spend_rate * chosen.cost,
                0.0,
                1.0,
            )
        )
        new_latency_budget = float(
            np.clip(
                pre_latency_budget + self.task_cfg.latency_recovery - self.task_cfg.latency_spend_rate * chosen.latency,
                0.0,
                1.0,
            )
        )
        new_risk_tolerance = float(
            np.clip(
                pre_risk_tolerance
                + self.task_cfg.risk_tolerance_recovery * (1.0 - float(aux["risk_exposure"]))
                - self.task_cfg.risk_tolerance_decay * float(aux["risk_exposure"]),
                0.0,
                1.0,
            )
        )

        new_trust = self._update_trust(
            pre_trust=pre_trust,
            y_true=y_t,
            confidence_score=action.confidence_score,
            calibration_target=float(aux["calibration_target"]),
            feedback_volatility=pre_volatility,
            saturation_triggered=saturation_triggered,
            resource_pressure=float(aux["resource_pressure"]),
            regime=new_regime,
            latent_vol=new_latent_vol,
        )

        new_trust_collapsed = bool(
            self.hidden.trust_collapsed
            or (
                self.task_cfg.trust_collapse_threshold >= 0.0
                and new_trust < self.task_cfg.trust_collapse_threshold
            )
        )
        new_risk_history = list(self.hidden.risk_history)
        new_risk_history.append(float(aux["risk_exposure"]))
        sustained_risk, risk_window_size = _rolling_mean(new_risk_history, 3)

        new_diversity_history = list(self.hidden.diversity_history)
        new_diversity_history.append(float(aux["diversity_pressure"]))
        sustained_diversity, diversity_window_size = _rolling_mean(new_diversity_history, 4)

        new_risk_collapsed = bool(
            self.hidden.risk_collapsed
            or (
                self.task_cfg.risk_collapse_threshold < 1.0
                and risk_window_size >= 3
                and sustained_risk > self.task_cfg.risk_collapse_threshold
            )
        )
        new_diversity_collapsed = bool(
            self.hidden.diversity_collapsed
            or saturation_triggered
            or (
                self.task_cfg.diversity_collapse_threshold < 1.0
                and diversity_window_size >= 4
                and sustained_diversity > self.task_cfg.diversity_collapse_threshold
            )
        )

        risk_noise_floor = float(self.hidden.risk_noise_floor)
        volatility_floor = float(self.hidden.volatility_floor)
        if new_risk_collapsed:
            risk_noise_floor = max(risk_noise_floor, self.task_cfg.risk_noise_boost)
            volatility_floor = max(volatility_floor, self.task_cfg.collapse_volatility_floor)

        new_p = update_patience(pre_p, y_t)

        z_arr = np.asarray(pre_z, dtype=float)
        m_arr = np.asarray(pre_m, dtype=float)
        phi_t, psi_t = self._drift_targets(z_arr, m_arr, self.hidden.task_id)
        forced_conflict = (
            self.hidden.task_id == "task_3"
            and self.hidden.turn == drift_trigger_turn
            and self.hidden.drift_turn is None
        )
        drift_mult = self.task_cfg.regime_drift_multiplier[new_regime] * (1.0 + 0.35 * new_latent_vol)
        noise_mult = self.task_cfg.regime_noise_multiplier[new_regime] * new_latent_vol

        if forced_conflict:
            z_target = psi_t
            base_rate = max(0.72, min(0.95, self.task_cfg.kappa * drift_mult + 0.30))
        elif new_p > self.task_cfg.tau:
            z_target = phi_t
            base_rate = self.task_cfg.mu * (0.80 + 0.40 * new_trust)
        else:
            z_target = psi_t
            base_rate = min(0.95, self.task_cfg.kappa + self.task_cfg.drift_trust_boost * (1.0 - new_trust))

        step_rate = float(
            np.clip(
                base_rate * drift_mult,
                0.02 if new_p > self.task_cfg.tau and not forced_conflict else 0.05,
                0.98,
            )
        )
        noise_scale = 0.015 + 0.05 * noise_mult
        new_z = (1.0 - step_rate) * z_arr + step_rate * np.asarray(z_target, dtype=float)
        new_z = new_z + self.rng.normal(0.0, noise_scale, size=K)
        new_z = _normalize(new_z)
        if forced_conflict:
            self.hidden.drift_turn = self.hidden.turn

        post_argmax_z = int(np.argmax(new_z))
        drift_happened = (post_argmax_z != pre_argmax_z)
        if drift_happened and self.hidden.drift_turn is None:
            self.hidden.drift_turn = self.hidden.turn

        alignment = float(np.dot(new_z, m_arr))
        new_chi = float(np.clip(pre_chi + GLOBAL.alpha_chi * (alignment - GLOBAL.theta_chi), 0.0, 1.0))
        if new_trust_collapsed or new_risk_collapsed:
            new_regime = max(new_regime, 3)
            new_latent_vol = max(new_latent_vol, 0.58)
        elif new_diversity_collapsed or saturation_triggered:
            new_regime = max(new_regime, 2)
            new_latent_vol = max(new_latent_vol, 0.40)
        elif drift_happened:
            new_regime = max(new_regime, 1)
            new_latent_vol = max(new_latent_vol, 0.22)

        platform_gain = float(
            np.clip(
                0.45 * observed_feedback
                + 0.25 * new_trust
                + 0.20 * new_p
                + 0.10 * new_p,
                0.0,
                1.0,
            )
        )
        self.latest_reward_breakdown.platform_gain = float(round(platform_gain, 6))

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
                trust_before=float(pre_trust),
                trust_after=float(new_trust),
                observed_feedback=float(observed_feedback),
                feedback_volatility=float(pre_volatility),
                repetition_pressure=float(aux["repetition_pressure"]),
                novelty_violation=float(aux["novelty_violation"]),
                resource_pressure=float(aux["resource_pressure"]),
                risk_exposure=float(aux["risk_exposure"]),
                diversity_pressure=float(aux["diversity_pressure"]),
                platform_gain=platform_gain,
                budget_remaining=float(pre_budget_remaining),
                risk_tolerance=float(pre_risk_tolerance),
                latency_budget=float(pre_latency_budget),
                calibration_target=float(aux["calibration_target"]),
                slot_type=chosen.slot_type,
                saturated=bool(saturated_now),
                trust_collapsed=new_trust_collapsed,
                risk_collapsed=new_risk_collapsed,
                diversity_collapsed=new_diversity_collapsed,
                regime=new_regime,
                latent_vol=float(new_latent_vol),
                exploration_flag=action.exploration_flag,
                confidence_score=action.confidence_score,
            )
        )

        self.hidden.history_item_ids.append(chosen.item_id)
        self.hidden.history_category_ids.append(chosen.category_id)
        self.hidden.history_topic_vectors.append([float(x) for x in chosen_x])

        fb_bucket = feedback_bucket(observed_feedback)
        self.recent_interactions.append(
            RecentInteraction(
                turn_id=self.hidden.turn,
                item_id=chosen.item_id,
                category_id=chosen.category_id,
                category_name=chosen.category_name,
                confidence_score=action.confidence_score,
                exploration_flag=action.exploration_flag,
                reward=reward_value,
                satisfaction_proxy=float(observed_feedback),
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
        self.hidden.trust = float(new_trust)
        self.hidden.budget_remaining = float(new_budget_remaining)
        self.hidden.risk_tolerance = float(new_risk_tolerance)
        self.hidden.latency_budget = float(new_latency_budget)
        self.hidden.feedback_history = [float(x) for x in new_feedback_history]
        self.hidden.risk_history = [float(x) for x in new_risk_history]
        self.hidden.diversity_history = [float(x) for x in new_diversity_history]
        self.hidden.trust_collapsed = bool(new_trust_collapsed)
        self.hidden.risk_collapsed = bool(new_risk_collapsed)
        self.hidden.diversity_collapsed = bool(new_diversity_collapsed)
        self.hidden.risk_noise_floor = float(risk_noise_floor)
        self.hidden.volatility_floor = float(volatility_floor)
        self.hidden.regime = int(new_regime)
        self.hidden.latent_vol = float(new_latent_vol)

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
            "observed_feedback": float(round(observed_feedback, 6)),
            "feedback_volatility": float(round(post_volatility, 6)),
            "trust_signal": float(round(new_trust, 6)),
            "budget_remaining": float(round(new_budget_remaining, 6)),
            "risk_tolerance": float(round(new_risk_tolerance, 6)),
            "latency_budget": float(round(new_latency_budget, 6)),
            "platform_gain": float(round(platform_gain, 6)),
            "risk_exposure": float(round(aux["risk_exposure"], 6)),
            "diversity_pressure": float(round(aux["diversity_pressure"], 6)),
            "resource_pressure": float(round(aux["resource_pressure"], 6)),
            "trust_collapsed": new_trust_collapsed,
            "risk_collapsed": new_risk_collapsed,
            "diversity_collapsed": new_diversity_collapsed,
            "regime": int(new_regime),
            "regime_name": REGIME_NAMES[int(new_regime)],
            "latent_vol": float(round(new_latent_vol, 6)),
            "latent_dud": is_dud,
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
                trust_level=self.hidden.trust,
                budget_remaining=self.hidden.budget_remaining,
                risk_tolerance=self.hidden.risk_tolerance,
                latency_budget=self.hidden.latency_budget,
                diversity_collapsed=self.hidden.diversity_collapsed,
                regime=self.hidden.regime,
                latent_vol=self.hidden.latent_vol,
            )
            obs = self._build_observation(session_feedback_signal=float(observed_feedback))
        else:
            self.candidate_items = []
            self._ensure_audit_results()
            self.final_breakdown = final_grade(
                self.trajectory,
                self.hidden.task_id,
                self.audit_results,
            )
            if self.final_breakdown.recovered_turn is not None:
                self.hidden.recovered_turn = self.final_breakdown.recovered_turn
            obs = self._build_observation(session_feedback_signal=float(observed_feedback))
            info["final_grade"] = self.final_breakdown.model_dump()
            if self.audit_results:
                info["counterfactual_audits"] = [
                    audit.model_dump() for audit in self.audit_results
                ]

        return StepResult(
            observation=obs,
            reward=float(round(reward_value, 6)),
            done=done,
            info=info,
        )

    def current_grade(self) -> FinalGradeBreakdown:
        if self.hidden is None:
            raise RuntimeError("Environment not initialized.")
        self._ensure_audit_results()
        return final_grade(self.trajectory, self.hidden.task_id, self.audit_results)
