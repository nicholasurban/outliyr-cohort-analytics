from __future__ import annotations

import base64
import io

from pypdf import PdfReader

from src.engine import analyze_request
from src.schemas import AnalyzeRequest


def test_rendered_report_is_valid_multi_page_pdf() -> None:
    payload = {
        "trial": {
            "sponsored_trial_id": 77,
            "title": "Magnesium Sleep Trial",
            "sponsor_name": "Acme",
            "protocol_hash": "abc123",
            "primary_endpoint": "sleep_score",
        },
        "endpoints": [
            {"key": "sleep_score", "label": "Sleep Score", "direction": "higher_better", "role": "primary"},
            {"key": "rhr_bpm", "label": "Resting Heart Rate", "direction": "lower_better", "role": "secondary"},
        ],
        "participants": [
            {
                "participant_id": "p1",
                "status": "completed",
                "verified_compliance_pct": 92,
                "endpoints": {
                    "sleep_score": {"baseline_values": [70, 72], "intervention_values": [80, 82], "daily_values": [76, 78, 82]},
                    "rhr_bpm": {"baseline_values": [60, 61], "intervention_values": [57, 58], "daily_values": [59, 58, 57]},
                },
            },
            {
                "participant_id": "p2",
                "status": "completed",
                "verified_compliance_pct": 80,
                "endpoints": {
                    "sleep_score": {"baseline_values": [65, 66], "intervention_values": [70, 72], "daily_values": [67, 70, 72]},
                    "rhr_bpm": {"baseline_values": [65, 64], "intervention_values": [63, 62], "daily_values": [64, 63, 62]},
                },
            },
        ],
        "concurrent_interventions": [],
        "adverse_events": [],
    }
    report = analyze_request(AnalyzeRequest.model_validate(payload))
    pdf_bytes = base64.b64decode(report["pdf_base64"])

    assert pdf_bytes.startswith(b"%PDF")
    assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) >= 5
