from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from .engine import ENGINE_VERSION, analyze_request
from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="Outliyr Sponsored Trial Analytics", version=ENGINE_VERSION)


def _allowed_ip(request: Request) -> bool:
    allowed = os.getenv("OUTLIYR_ANALYTICS_ALLOWED_IPS", "").strip()
    if not allowed:
        return True
    candidates = {item.strip() for item in allowed.split(",") if item.strip()}
    client_ip = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return client_ip in candidates or forwarded in candidates


def require_service_auth(
    request: Request,
    x_outliyr_analytics_secret: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("OUTLIYR_ANALYTICS_TOKEN", "") or os.getenv("OUTLIYR_ANALYTICS_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="service_secret_not_configured")
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if not bearer and not x_outliyr_analytics_secret:
        raise HTTPException(status_code=401, detail="missing_service_secret")
    legacy = x_outliyr_analytics_secret or ""
    bearer_ok = bool(bearer) and hmac.compare_digest(bearer, expected)
    legacy_ok = bool(legacy) and hmac.compare_digest(legacy, expected)
    if not bearer_ok and not legacy_ok:
        raise HTTPException(status_code=403, detail="invalid_service_secret")
    if not _allowed_ip(request):
        raise HTTPException(status_code=403, detail="ip_not_allowed")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "engine_version": ENGINE_VERSION}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": ENGINE_VERSION, "engine_version": ENGINE_VERSION}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, _: None = Depends(require_service_auth)) -> dict:
    return analyze_request(payload)


@app.post("/v1/cohort-report", response_model=AnalyzeResponse)
def cohort_report_v1(payload: AnalyzeRequest, _: None = Depends(require_service_auth)) -> dict:
    return analyze_request(payload)


@app.post("/cohort-report", response_model=AnalyzeResponse)
def cohort_report(payload: AnalyzeRequest, _: None = Depends(require_service_auth)) -> dict:
    return analyze_request(payload)
