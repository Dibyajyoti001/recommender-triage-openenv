from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def reset(self, task_id: str = "task_1", session_id: str = "default", seed: Optional[int] = None) -> Dict[str, Any]:
        r = self.http.post(f"{self.base_url}/reset", params={"task_id": task_id, "session_id": session_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        r = self.http.post(f"{self.base_url}/step", params={"session_id": session_id}, json=action)
        r.raise_for_status()
        return r.json()

    def state(self, session_id: str = "default") -> Dict[str, Any]:
        r = self.http.get(f"{self.base_url}/state", params={"session_id": session_id})
        r.raise_for_status()
        return r.json()

    def tasks(self) -> Dict[str, Any]:
        r = self.http.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()

    def grader(self, session_id: str = "default") -> Dict[str, Any]:
        r = self.http.get(f"{self.base_url}/grader", params={"session_id": session_id})
        r.raise_for_status()
        return r.json()

    def baseline(self, num_runs: int = 3) -> Dict[str, Any]:
        r = self.http.get(f"{self.base_url}/baseline", params={"num_runs": num_runs})
        r.raise_for_status()
        return r.json()