# outliyr-cohort-analytics

FastAPI microservice that computes cohort statistics + renders sponsor PDF reports for Outliyr Sponsored Trials.

## Status

Scaffold only. Codex builds this out per `docs/superpowers/plans/2026-04-30-spt-analytics-service-and-pdf.md` in the `claude-skills` repo.

## Stack

- Python 3.11
- FastAPI + uvicorn
- scipy, statsmodels (paired t-tests, mixed-effects models, BH-FDR)
- pandas, numpy
- reportlab (sponsor PDF rendering)
- httpx (callback to outliyr.com)

## Deploy

- Hosted on auto.outliyr.com via Coolify (Docker app).
- Coolify auto-deploys on push to `main`.
- Required env vars (set in Coolify dashboard, NOT committed): `OUTLIYR_ANALYTICS_TOKEN`, `OUTLIYR_CALLBACK_URL`, `LOG_LEVEL`.
- Same `OUTLIYR_ANALYTICS_TOKEN` is mirrored in outliyr.com `wp-config.php`.

## API

`POST /v1/cohort-report` (Bearer auth) -> queues a report job. WordPress posts trial enrollment + check-in data; service responds 202 with job id; on completion service POSTs PDF + JSON back to `OUTLIYR_CALLBACK_URL` with HMAC-signed body.

`GET /healthz` -> liveness for Coolify healthcheck.

## Auth

Bearer token in `Authorization` header. Token verified against `OUTLIYR_ANALYTICS_TOKEN` env. Constant-time compare. Reject any request without it. Callbacks back to outliyr.com sign the body with the same token via HMAC-SHA256 in `X-Outliyr-Signature` header.
