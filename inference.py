from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Environment server base URL
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

# Logging metadata
BENCHMARK = os.getenv("BENCHMARK", "recommendation-policy-triage")
SESSION_ID = os.getenv("SESSION_ID", "default")

MAX_STEPS = 20
TIMEOUT = 60.0

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
- Output JSON only, with no extra text.
""".strip()


def require_env() -> None:
    missing: List[str] = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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

        if score > best_score:
            best_score = score
            best = item

    explore = bool(
        obs.get("memory_confidence", 0.0) < 0.55
        or obs.get("repetition_pressure_bucket") == "high"
        or best.get("freshness") == "novel"
    )

    return {
        "recommended_item_id": int(best["item_id"]),
        "exploration_flag": explore,
        "confidence_score": 0.80 if not explore else 0.65,
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
        "candidate_items": [
            {
                "item_id": c.get("item_id"),
                "category_id": c.get("category_id"),
                "category_name": c.get("category_name"),
                "quality": c.get("quality"),
                "engagement": c.get("engagement"),
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
        rec_id = int(data["recommended_item_id"])
        explore = bool(data["exploration_flag"])
        conf = max(0.0, min(1.0, float(data["confidence_score"])))
        return {
            "recommended_item_id": rec_id,
            "exploration_flag": explore,
            "confidence_score": conf,
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
        content = response.choices[0].message.content or ""
        return parse_action(content, obs)
    except Exception:
        return fallback_action(obs)


def action_to_log_string(action: Dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=False)


def run_task(task_id: str, client: OpenAI, http: httpx.Client) -> Dict[str, Any]:
    reset_resp = http.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id, "session_id": SESSION_ID},
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    done = False
    step_count = 0
    rewards: List[float] = []
    last_info: Dict[str, Any] = {}
    success = False
    score = 0.0

    task_name = obs.get("task_name", task_id)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME or "")

    try:
        while not done and step_count < MAX_STEPS:
            action = llm_action(client, obs)
            action_str = action_to_log_string(action)

            error_msg: Optional[str] = None
            reward = 0.0

            try:
                step_resp = http.post(
                    f"{ENV_BASE_URL}/step",
                    params={"session_id": SESSION_ID},
                    json=action,
                )
                step_resp.raise_for_status()
                data = step_resp.json()

                obs = data["observation"]
                done = bool(data["done"])
                reward = float(data.get("reward", 0.0) or 0.0)
                last_info = data.get("info", {})
            except Exception as exc:
                done = True
                error_msg = str(exc)

            step_count += 1
            rewards.append(reward)
            log_step(
                step=step_count,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        score_payload: Optional[Dict[str, Any]] = None
        try:
            grade_resp = http.get(
                f"{ENV_BASE_URL}/grader",
                params={"session_id": SESSION_ID},
            )
            grade_resp.raise_for_status()
            score_payload = grade_resp.json()
        except Exception:
            final_grade = last_info.get("final_grade")
            if isinstance(final_grade, dict) and "final_score" in final_grade:
                score_payload = {
                    "score": final_grade["final_score"],
                    "breakdown": final_grade,
                }

        if score_payload is not None:
            score = float(score_payload.get("score", 0.0))
            success = score > 0.0
        else:
            success = False
            score = 0.0

        return {
            "task_id": task_id,
            "steps": step_count,
            "score": score,
            "rewards": rewards,
            "success": success,
        }

    finally:
        log_end(success=success, steps=step_count, score=score, rewards=rewards)


def main() -> None:
    require_env()

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    with httpx.Client(timeout=TIMEOUT) as http:
        for task_id in ["task_1", "task_2", "task_3"]:
            run_task(task_id, client, http)
            


if __name__ == "__main__":
    main()
