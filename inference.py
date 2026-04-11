from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

BENCHMARK = os.getenv("BENCHMARK", "recommendation-policy-triage")
SESSION_ID = os.getenv("SESSION_ID", "default")

MAX_STEPS = 20
TIMEOUT = 60.0
DEFAULT_TASK_IDS = ("task_1", "task_2", "task_3", "task_4", "task_5")

SYSTEM_PROMPT = """
You control a recommendation-policy environment.

You will receive a structured observation for one step of a user session.
Choose exactly one candidate item and decide:
- whether this step is exploratory
- how confident you are

Return ONLY valid JSON with this exact schema:
{
  "recommended_item_id": <int>,
  "exploration_flag": <true_or_false>,
  "confidence_score": <float_between_0_and_1>
}

Guidelines:
- Use memory summary when memory confidence is strong.
- Avoid excessive repetition when repetition pressure is high.
- In conflict cases, recent interaction signals may override stale memory.
- Lower confidence when feedback volatility is high or trust is weak.
- Respect visible budget, risk, and latency constraints when choosing items.
- Protect long-term trust and engagement, not just short-term fit.
- Output JSON only, with no extra text.
""".strip()


def require_env() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    candidates = obs.get("candidate_items", [])
    if not candidates:
        return {
            "recommended_item_id": 0,
            "exploration_flag": False,
            "confidence_score": 0.0,
        }

    memory_summary = obs.get("memory_summary", {}) or {}
    top_categories = memory_summary.get("top_categories", []) or []
    top_memory_cat = top_categories[0] if top_categories else 0

    repetition_counts = obs.get("repetition_counts", []) or []

    best = candidates[0]
    best_score = float("-inf")

    for item in candidates:
        cat = int(item.get("category_id", 0))
        quality = float(item.get("quality", 0.0))
        engagement = float(item.get("engagement", 0.0))
        cost = float(item.get("cost", 0.0))
        risk = float(item.get("risk", 0.0))
        latency = float(item.get("latency", 0.0))
        freshness = item.get("freshness", "fresh")

        score = 0.60 * quality + 0.20 * engagement

        if cat == top_memory_cat:
            score += 0.10

        if cat < len(repetition_counts):
            score -= 0.06 * repetition_counts[cat]

        if freshness == "fresh":
            score += 0.04
        elif freshness == "novel":
            score += 0.02
        elif freshness == "stale":
            score -= 0.02

        score -= 0.12 * cost / max(float(obs.get("budget_remaining", 1.0)), 0.10)
        score -= 0.14 * risk / max(float(obs.get("risk_tolerance", 1.0)), 0.10)
        score -= 0.10 * latency / max(float(obs.get("latency_budget", 1.0)), 0.10)

        if score > best_score:
            best_score = score
            best = item

    explore = bool(
        obs.get("memory_confidence", 0.0) < 0.55
        or obs.get("repetition_pressure_bucket") == "high"
        or best.get("freshness") == "novel"
        or obs.get("trust_signal", 1.0) < 0.50
    )

    confidence = 0.88 - 2.0 * float(obs.get("feedback_volatility", 0.0))
    if explore:
        confidence -= 0.12
    if float(obs.get("trust_signal", 1.0)) < 0.45:
        confidence -= 0.08
    confidence = max(0.20, min(0.92, confidence))

    return {
        "recommended_item_id": int(best["item_id"]),
        "exploration_flag": explore,
        "confidence_score": confidence,
    }


def build_user_prompt(obs: Dict[str, Any]) -> str:
    compact = {
        "task_id": obs.get("task_id"),
        "task_name": obs.get("task_name"),
        "turn_id": obs.get("turn_id"),
        "max_turns": obs.get("max_turns"),
        "memory_summary": obs.get("memory_summary"),
        "recent_interactions": obs.get("recent_interactions"),
        "repetition_counts": obs.get("repetition_counts"),
        "repetition_pressure_bucket": obs.get("repetition_pressure_bucket"),
        "memory_confidence_bucket": obs.get("memory_confidence_bucket"),
        "memory_confidence": obs.get("memory_confidence"),
        "session_feedback_signal": obs.get("session_feedback_signal"),
        "feedback_volatility": obs.get("feedback_volatility"),
        "trust_signal": obs.get("trust_signal"),
        "engagement_signal": obs.get("engagement_signal"),
        "budget_remaining": obs.get("budget_remaining"),
        "risk_tolerance": obs.get("risk_tolerance"),
        "latency_budget": obs.get("latency_budget"),
        "engagement_bucket": obs.get("engagement_bucket"),
        "candidate_items": [
            {
                "item_id": c.get("item_id"),
                "category_id": c.get("category_id"),
                "category_name": c.get("category_name"),
                "quality": c.get("quality"),
                "engagement": c.get("engagement"),
                "cost": c.get("cost"),
                "risk": c.get("risk"),
                "latency": c.get("latency"),
                "freshness": c.get("freshness"),
                "slot_type": c.get("slot_type"),
            }
            for c in obs.get("candidate_items", [])
        ],
    }
    return json.dumps(compact, ensure_ascii=False)


def parse_action(content: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = json.loads(content)
        return {
            "recommended_item_id": int(data["recommended_item_id"]),
            "exploration_flag": bool(data["exploration_flag"]),
            "confidence_score": max(0.0, min(1.0, float(data["confidence_score"]))),
        }
    except Exception:
        return fallback_action(obs)


def llm_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
        )
        return parse_action(response.choices[0].message.content or "", obs)
    except Exception:
        return fallback_action(obs)


def action_to_log_string(action: Dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=False)


def reset_with_retry(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(1, 11):
        try:
            response = http.post(
                f"{ENV_BASE_URL}/reset",
                params={"task_id": task_id, "session_id": SESSION_ID},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == 10:
                break
            time.sleep(min(2**attempt, 20))

    raise RuntimeError(f"Failed to reset environment for {task_id}") from last_error


def fetch_task_ids(http: httpx.Client) -> List[str]:
    try:
        response = http.get(f"{ENV_BASE_URL}/tasks")
        response.raise_for_status()
        payload = response.json()
        tasks = payload.get("tasks", [])
        task_ids = [str(task["task_id"]) for task in tasks if "task_id" in task]
        if task_ids:
            return task_ids
    except Exception:
        pass
    return list(DEFAULT_TASK_IDS)


def run_task(task_id: str, client: OpenAI, http: httpx.Client) -> Dict[str, Any]:
    done = False
    step_count = 0
    rewards: List[float] = []
    last_info: Dict[str, Any] = {}
    success = False
    score = 0.0
    obs: Dict[str, Any] = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME or "")

    try:
        obs = reset_with_retry(http, task_id)

        while not done and step_count < MAX_STEPS:
            action = llm_action(client, obs)

            try:
                response = http.post(
                    f"{ENV_BASE_URL}/step",
                    params={"session_id": SESSION_ID},
                    json=action,
                )
                response.raise_for_status()
                data = response.json()

                obs = data["observation"]
                done = bool(data["done"])
                reward = float(data.get("reward", 0.0))
                last_info = data.get("info", {})
                error_msg = None
            except Exception as exc:
                done = True
                reward = 0.0
                error_msg = str(exc)

            step_count += 1
            rewards.append(reward)

            log_step(step_count, action_to_log_string(action), reward, done, error_msg)

        try:
            response = http.get(
                f"{ENV_BASE_URL}/grader",
                params={"session_id": SESSION_ID},
            )
            response.raise_for_status()
            score_payload = response.json()
        except Exception:
            score_payload = last_info.get("final_grade", {})

        score = float(score_payload.get("score", score_payload.get("final_score", 0.0)))
        success = score > 0.0

        return {
            "task_id": task_id,
            "steps": step_count,
            "score": score,
            "rewards": rewards,
            "success": success,
        }

    finally:
        log_end(success, step_count, score, rewards)


def main() -> None:
    require_env()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with httpx.Client(timeout=TIMEOUT) as http:
        for task_id in fetch_task_ids(http):
            run_task(task_id, client, http)


if __name__ == "__main__":
    main()
