from __future__ import annotations

from typing import Any

from .engine import _render_pdf


def render_pdf(report: dict[str, Any], report_hash: str) -> bytes:
    return _render_pdf(report, report_hash)
