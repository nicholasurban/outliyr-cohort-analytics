# outliyr-cohort-analytics

FastAPI microservice that computes cohort statistics + renders sponsor PDF reports for Outliyr Sponsored Trials.

## Status

Production service for `docs/superpowers/plans/2026-04-30-spt-analytics-service-and-pdf.md` in the `claude-skills` repo.

## Stack

- Python 3.11
- FastAPI + uvicorn
- scipy, statsmodels (paired t-tests, mixed-effects models, BH-FDR)
- pandas, numpy
- matplotlib + PDF backend (sponsor PDF rendering)

## Deploy

- Hosted on auto.outliyr.com via Coolify (Docker app).
- Coolify auto-deploys on push to `main`.
- Required env vars (set in Coolify dashboard, NOT committed): `OUTLIYR_ANALYTICS_TOKEN`, `OUTLIYR_CALLBACK_URL`, `LOG_LEVEL`.
- Same `OUTLIYR_ANALYTICS_TOKEN` is mirrored in outliyr.com `wp-config.php`.

## API

`POST /v1/cohort-report` (Bearer auth) -> returns a versioned cohort analytics JSON payload with `pdf_base64`.

`POST /cohort-report` and `POST /v1/analyze` remain as compatibility aliases for older WordPress clients.

`GET /healthz` -> liveness for Coolify healthcheck.

`GET /health` -> human-readable health payload.

## Auth

Bearer token in `Authorization` header. Token verified against `OUTLIYR_ANALYTICS_TOKEN` env with constant-time comparison. Requests without the bearer token are rejected.
