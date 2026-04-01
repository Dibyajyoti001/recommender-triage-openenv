from __future__ import annotations

from typing import Dict, List

from .tasks import CATEGORY_NAMES


CATEGORY_ID_TO_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(CATEGORY_NAMES)}
CATEGORY_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in CATEGORY_ID_TO_NAME.items()}

DEFAULT_SEEDS: Dict[str, int] = {
    "task_1": 101,
    "task_2": 202,
    "task_3": 303,
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "task_1": "Stable Preference Exploitation",
    "task_2": "Repetition Fatigue Control",
    "task_3": "Memory vs Live Signal Conflict",
}