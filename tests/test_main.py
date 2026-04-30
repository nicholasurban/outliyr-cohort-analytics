from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _payload() -> dict:
    return {
        "trial": {
            "sponsored_trial_id": 1,
            "title": "Cold Plunge Demo",
            "sponsor_name": "Outliyr",
            "primary_endpoint": "sleep_score",
        },
        "endpoints": [
            {"key": "sleep_score", "label": "Sleep Score", "direction": "higher_better", "role": "primary"}
        ],
        "participants": [],
        "concurrent_interventions": [],
        "adverse_events": [],
    }


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cohort_report_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("OUTLIYR_ANALYTICS_TOKEN", "test-token")

    response = client.post("/v1/cohort-report", json=_payload())

    assert response.status_code == 401


def test_cohort_report_rejects_wrong_token(monkeypatch) -> None:
    monkeypatch.setenv("OUTLIYR_ANALYTICS_TOKEN", "test-token")

    response = client.post("/v1/cohort-report", json=_payload(), headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 403
