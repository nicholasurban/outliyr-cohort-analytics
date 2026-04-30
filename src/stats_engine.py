from __future__ import annotations

from typing import Any

from .engine import analyze_request
from .schemas import AnalyzeRequest


def run_cohort_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_request(AnalyzeRequest.model_validate(payload))
