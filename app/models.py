from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


CategoryName = Literal[
    "Romance",
    "Comedy",
    "Drama",
    "Thriller",
    "Action",
    "Documentary",
    "Crime",
    "SciFi",
    "News",
    "Lifestyle",
]
FeedbackBucket = Literal["low", "neutral", "positive", "strong_positive"]
PressureBucket = Literal["low", "medium", "high"]
ConfidenceBucket = Literal["weak", "moderate", "strong"]
FreshnessBucket = Literal["fresh", "stale", "novel"]
EngagementBucket = Literal["fragile", "stable", "engaged"]


class CandidateItem(BaseModel):
    """
    A single candidate shown to the agent.

    `topic_vector` is the canonical mixed-topic sparse representation over the
    fixed global basis from tasks.py -> CATEGORY_NAMES.
    `category_id` / `category_name` are the primary topic only (argmax).
    """

    item_id: int
    title: str
    category_id: int = Field(ge=0)
    category_name: CategoryName
    quality: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0, default=0.0)
    risk: float = Field(ge=0.0, le=1.0, default=0.0)
    latency: float = Field(ge=0.0, le=1.0, default=0.0)
    freshness: FreshnessBucket
    style_vector: List[float]
    slot_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    topic_vector: List[float] = Field(default_factory=list)


class RecentInteraction(BaseModel):
    turn_id: int
    item_id: int
    category_id: int
    category_name: CategoryName
    confidence_score: float = Field(ge=0.0, le=1.0)
    exploration_flag: bool
    reward: float
    satisfaction_proxy: float = Field(ge=0.0, le=1.0)
    feedback_bucket: FeedbackBucket


class MemorySummary(BaseModel):
    top_categories: List[int]
    top_category_names: List[CategoryName]
    coarse_scores: List[float]
    summary_text: str


class Observation(BaseModel):
    task_id: str
    task_name: str
    turn_id: int
    max_turns: int
    memory_summary: MemorySummary
    recent_interactions: List[RecentInteraction]
    candidate_items: List[CandidateItem]
    repetition_counts: List[int]
    repetition_pressure_bucket: PressureBucket
    memory_confidence_bucket: ConfidenceBucket
    memory_confidence: float = Field(ge=0.0, le=1.0)
    session_feedback_signal: float = Field(ge=0.0, le=1.0)
    feedback_volatility: float = Field(ge=0.0, le=1.0, default=0.0)
    trust_signal: float = Field(ge=0.0, le=1.0, default=0.0)
    engagement_signal: float = Field(ge=0.0, le=1.0, default=0.0)
    budget_remaining: float = Field(ge=0.0, le=1.0, default=1.0)
    risk_tolerance: float = Field(ge=0.0, le=1.0, default=1.0)
    latency_budget: float = Field(ge=0.0, le=1.0, default=1.0)
    engagement_bucket: EngagementBucket = "stable"
    done_hint: str = ""


class Action(BaseModel):
    recommended_item_id: int
    exploration_flag: bool
    confidence_score: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class RewardBreakdown(BaseModel):
    relevance: float
    fatigue_cost: float
    novelty_violation: float
    unnecessary_exploration: float
    confidence_penalty: float
    trust_bonus: float = 0.0
    calibration_target: float = 1.0
    calibration_gap: float = 0.0
    resource_pressure: float = 0.0
    risk_exposure: float = 0.0
    diversity_pressure: float = 0.0
    platform_gain: float = 0.0
    raw_reward: float
    clipped_reward: float
    satisfaction_proxy: float
    repetition_pressure: float = 0.0
    alignment: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class HiddenState(BaseModel):
    """
    Hidden simulator state. This should only be exposed through /state for debugging.
    """

    task_id: str
    turn: int
    m: List[float]          # long-term memory preference over fixed 10-topic basis
    z: List[float]          # live session intent over fixed 10-topic basis
    H_topic: List[float]    # EMA history pressure over fixed 10-topic basis
    F_topic: List[float]    # fatigue accumulator over fixed 10-topic basis
    history_topic_vectors: List[List[float]] = Field(default_factory=list)

    # Compatibility fields preserved for earlier code / graders / inspection
    F_cat: List[float]
    F_item: Dict[int, float]
    nu: float
    p: float
    chi: float
    history_item_ids: List[int]
    history_category_ids: List[int]
    drift_turn: Optional[int] = None
    recovered_turn: Optional[int] = None
    last_feedback_bucket: FeedbackBucket = "neutral"
    trust: float = 0.0
    budget_remaining: float = 1.0
    risk_tolerance: float = 1.0
    latency_budget: float = 1.0
    feedback_history: List[float] = Field(default_factory=list)
    risk_history: List[float] = Field(default_factory=list)
    diversity_history: List[float] = Field(default_factory=list)
    saturation_turn: Optional[int] = None
    trust_collapsed: bool = False
    risk_collapsed: bool = False
    diversity_collapsed: bool = False
    risk_noise_floor: float = 0.0
    volatility_floor: float = 0.0
    rng_seed: int = 0


class CounterfactualAuditResult(BaseModel):
    turn_id: int
    fragility: float = Field(ge=0.0, le=1.0)
    chosen_value: float = Field(ge=0.0, le=1.0)
    safe_value: float = Field(ge=0.0, le=1.0)
    risky_value: float = Field(ge=0.0, le=1.0)
    safe_advantage: float
    risky_advantage: float
    audit_score: float = Field(ge=0.0, le=1.0)
    chosen_slot_type: str = ""
    safe_slot_type: str = ""
    risky_slot_type: str = ""


class FinalGradeBreakdown(BaseModel):
    satisfaction: float = Field(ge=0.0, le=1.0)
    diversity: float = Field(ge=0.0, le=1.0)
    adaptation: float = Field(ge=0.0, le=1.0)
    memory_use: float = Field(ge=0.0, le=1.0)
    trust: float = Field(ge=0.0, le=1.0, default=0.0)
    calibration: float = Field(ge=0.0, le=1.0, default=0.0)
    risk_safety: float = Field(ge=0.0, le=1.0, default=0.0)
    resource_efficiency: float = Field(ge=0.0, le=1.0, default=0.0)
    counterfactual_audit: float = Field(ge=0.0, le=1.0, default=0.0)
    audited_turns: int = Field(ge=0, default=0)
    collapse_gate: float = Field(ge=0.0, le=1.0, default=1.0)
    floor_gate: float = Field(ge=0.0, le=1.0, default=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    drift_turn: Optional[int] = None
    recovered_turn: Optional[int] = None
    task_weights: Dict[str, float] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_turns: int
    grader_weights: Dict[str, float]
    parameters: Dict[str, Any]


class TaskListResponse(BaseModel):
    tasks: List[TaskSpec]


class GraderResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: FinalGradeBreakdown


class EnvironmentState(BaseModel):
    """
    Full environment state for debugging.
    """

    hidden_state: HiddenState
    task_spec: TaskSpec
    trajectory_length: int
    latest_reward_breakdown: Optional[RewardBreakdown] = None
    audit_results: List[CounterfactualAuditResult] = Field(default_factory=list)
