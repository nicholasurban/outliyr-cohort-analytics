from __future__ import annotations

import base64
import io

from pypdf import PdfReader

from src.engine import analyze_request
from src.schemas import AnalyzeRequest
from tests.test_engine import SponsoredTrialAnalyticsEngineTest


def test_rendered_report_is_valid_multi_page_pdf() -> None:
    payload = SponsoredTrialAnalyticsEngineTest().sample_payload()
    report = analyze_request(AnalyzeRequest.model_validate(payload))
    pdf_bytes = base64.b64decode(report["pdf_base64"])

    assert pdf_bytes.startswith(b"%PDF")
    assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) >= 5
