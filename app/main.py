from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .graders import FinalGradeBreakdown
from .models import Action, EnvironmentState, GraderResponse, Observation, StepResult, TaskListResponse
from .simulator import RecommendationPolicyEnvironment
from .tasks import list_task_specs


app = FastAPI(
    title="Recommendation Policy Triage Environment",
    version="1.0.0",
    description=(
        "A complete OpenEnv-style environment for sequential recommendation policy "
        "under memory uncertainty, repetition fatigue, and intent drift."
    ),
)

_SESSIONS: Dict[str, RecommendationPolicyEnvironment] = {}
_DEFAULT_SESSION_ID = "default"


class BaselineResponse(BaseModel):
    task_scores: Dict[str, float]
    average_score: float


def _get_env(session_id: str = _DEFAULT_SESSION_ID) -> RecommendationPolicyEnvironment:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return env


def _balanced_heuristic_action(obs: Observation) -> Action:
    """
    Strong non-learning baseline from the frozen planner:
    favor relevance-like proxy, penalize repetition, and use memory confidence.
    Since hidden z is not exposed, this baseline uses available observation features only.
    """
    candidates = obs.candidate_items
    if not candidates:
        return Action(recommended_item_id=0, exploration_flag=False, confidence_score=0.0)

    top_memory_cat = obs.memory_summary.top_categories[0] if obs.memory_summary.top_categories else 0
    recent_majority: Optional[int] = None
    if obs.recent_interactions:
        cats = [x.category_id for x in obs.recent_interactions[-3:]]
        recent_majority = max(set(cats), key=cats.count)

    best_item = candidates[0]
    best_score = float("-inf")

    for item in candidates:
        repetition_pen = obs.repetition_counts[item.category_id] * 0.08
        memory_bonus = 0.12 if item.category_id == top_memory_cat else 0.0
        freshness_bonus = 0.06 if item.freshness == "fresh" else (0.08 if item.freshness == "novel" else -0.02)
        anti_repeat_bonus = 0.05 if recent_majority is not None and item.category_id != recent_majority else 0.0

        score = (
            0.55 * item.quality
            + 0.20 * item.engagement
            + memory_bonus
            + freshness_bonus
            + anti_repeat_bonus
            - repetition_pen
        )

        if score > best_score:
            best_score = score
            best_item = item

    explore = bool(
        obs.memory_confidence < 0.55
        or obs.repetition_pressure_bucket == "high"
        or best_item.freshness == "novel"
    )
    confidence = 0.80 if not explore else 0.65
    return Action(
        recommended_item_id=best_item.item_id,
        exploration_flag=explore,
        confidence_score=confidence,
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Recommendation Policy Triage Environment is running.",
    }


@app.get("/tasks", response_model=TaskListResponse)
def tasks_endpoint() -> TaskListResponse:
    return TaskListResponse(tasks=list_task_specs())


@app.post("/reset", response_model=Observation)
def reset_endpoint(
    task_id: str = Query(default="task_1"),
    session_id: str = Query(default=_DEFAULT_SESSION_ID),
    seed: Optional[int] = Query(default=None),
) -> Observation:
    env = RecommendationPolicyEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    _SESSIONS[session_id] = env
    return obs


@app.post("/step", response_model=StepResult)
def step_endpoint(
    action: Action,
    session_id: str = Query(default=_DEFAULT_SESSION_ID),
) -> StepResult:
    env = _get_env(session_id)
    return env.step(action)


@app.get("/state", response_model=EnvironmentState)
def state_endpoint(session_id: str = Query(default=_DEFAULT_SESSION_ID)) -> EnvironmentState:
    env = _get_env(session_id)
    return env.state()


@app.get("/grader", response_model=GraderResponse)
def grader_endpoint(session_id: str = Query(default=_DEFAULT_SESSION_ID)) -> GraderResponse:
    env = _get_env(session_id)
    breakdown: FinalGradeBreakdown = env.current_grade()
    return GraderResponse(score=breakdown.final_score, breakdown=breakdown)


@app.get("/baseline", response_model=BaselineResponse)
def baseline_endpoint(num_runs: int = Query(default=3, ge=1, le=20)) -> BaselineResponse:
    """
    Lightweight built-in baseline endpoint for quick validation.
    This does NOT replace root inference.py, but is useful for smoke validation.
    """
    task_scores: Dict[str, float] = {}

    for task_id in ["task_1", "task_2", "task_3"]:
        scores: List[float] = []
        for run_idx in range(num_runs):
            env = RecommendationPolicyEnvironment(seed=10_000 + run_idx)
            obs = env.reset(task_id=task_id, seed=20_000 + run_idx)

            done = False
            while not done:
                action = _balanced_heuristic_action(obs)
                result = env.step(action)
                obs = result.observation
                done = result.done

            grade = env.current_grade()
            scores.append(grade.final_score)

        task_scores[task_id] = round(sum(scores) / len(scores), 6)

    average_score = round(sum(task_scores.values()) / len(task_scores), 6)
    return BaselineResponse(task_scores=task_scores, average_score=average_score)
