from __future__ import annotations

import base64
import hashlib
import io
import json
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .schemas import AnalyzeRequest, Endpoint, Participant

ENGINE_VERSION = "engine_v1.0.0"
MIN_PER_PROTOCOL_COMPLIANCE = 75.0


def analyze_request(req: AnalyzeRequest) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    endpoint_results = [_endpoint_result(endpoint, req.participants) for endpoint in req.endpoints]
    raw_wilcoxon_p_values = [r["wilcoxon_p"] for r in endpoint_results if r["wilcoxon_p"] is not None]
    if raw_wilcoxon_p_values:
        adjusted = iter(multipletests(raw_wilcoxon_p_values, method="bonferroni")[1].tolist())
        for result in endpoint_results:
            result["bonferroni_p"] = float(next(adjusted)) if result["wilcoxon_p"] is not None else None
    else:
        for result in endpoint_results:
            result["bonferroni_p"] = None
    raw_paired_t_p_values = [r["paired_t_p"] for r in endpoint_results if r["paired_t_p"] is not None]
    if raw_paired_t_p_values:
        adjusted = iter(multipletests(raw_paired_t_p_values, method="fdr_bh")[1].tolist())
        for result in endpoint_results:
            result["bh_fdr_p"] = float(next(adjusted)) if result["paired_t_p"] is not None else None
    else:
        for result in endpoint_results:
            result["bh_fdr_p"] = None

    charts = {
        "waterfall": [_waterfall_chart(endpoint, req.participants) for endpoint in req.endpoints],
        "time_series": [_time_series_chart(endpoint, req.participants) for endpoint in req.endpoints],
    }
    report_without_pdf = {
        "engine_version": ENGINE_VERSION,
        "generated_at": generated_at,
        "trial": req.trial.model_dump(),
        "endpoint_results": endpoint_results,
        "dropout_sensitivity": _dropout_sensitivity(req.endpoints, req.participants),
        "subgroup_analysis": _subgroup_analysis(req.endpoints, req.participants),
        "charts": charts,
        "methodology": _methodology(req),
    }
    report_hash = hashlib.sha256(
        json.dumps(report_without_pdf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **report_without_pdf,
        "report_hash": report_hash,
        "pdf_base64": base64.b64encode(_render_pdf(report_without_pdf, report_hash)).decode("ascii"),
    }


def _endpoint_result(endpoint: Endpoint, participants: Sequence[Participant]) -> Dict[str, Any]:
    rows = _participant_deltas(endpoint, participants, per_protocol=False)
    pp_rows = _participant_deltas(endpoint, participants, per_protocol=True)
    deltas = np.array([row["oriented_delta"] for row in rows], dtype=float)
    pp_deltas = np.array([row["oriented_delta"] for row in pp_rows], dtype=float)
    baseline_values = np.array([row["baseline_mean"] for row in rows], dtype=float)
    intervention_values = np.array([row["intervention_mean"] for row in rows], dtype=float)
    ci_low, ci_high = _ci95(deltas)
    wilcoxon_p = _wilcoxon_p(deltas)
    threshold = 0.0 if endpoint.responder_threshold is None else float(endpoint.responder_threshold)
    responders = int(np.sum(deltas > threshold)) if deltas.size else 0
    return {
        "endpoint_key": endpoint.key,
        "label": endpoint.label,
        "role": endpoint.role,
        "direction": endpoint.direction,
        "n_itt": int(deltas.size),
        "n_per_protocol": int(pp_deltas.size),
        "baseline_mean": _mean_or_none(baseline_values),
        "baseline_sd": _sd_or_none(baseline_values),
        "intervention_mean": _mean_or_none(intervention_values),
        "intervention_sd": _sd_or_none(intervention_values),
        "itt_mean_delta": _round_or_none(float(np.mean(deltas))) if deltas.size else None,
        "itt_delta_sd": _sd_or_none(deltas),
        "per_protocol_mean_delta": _round_or_none(float(np.mean(pp_deltas))) if pp_deltas.size else None,
        "ci95": {"low": ci_low, "high": ci_high},
        "cohen_d": _cohen_dz(deltas),
        "paired_t_p": _paired_t_p(baseline_values, intervention_values),
        "bh_fdr_p": None,
        "wilcoxon_p": wilcoxon_p,
        "bonferroni_p": None,
        "responder_rate": _round_or_none(responders / deltas.size) if deltas.size else None,
        "responders": responders,
        "waterfall_points": [
            {"participant_id": row["participant_id"], "delta": _round_or_none(row["oriented_delta"])}
            for row in sorted(rows, key=lambda item: item["oriented_delta"])
        ],
    }


def _participant_deltas(endpoint: Endpoint, participants: Sequence[Participant], per_protocol: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for participant in participants:
        if per_protocol and not _is_per_protocol(participant):
            continue
        data = participant.endpoints.get(endpoint.key)
        if data is None or not data.baseline_values or not data.intervention_values:
            continue
        baseline = _safe_mean(data.baseline_values)
        intervention = _safe_mean(data.intervention_values)
        if baseline is None or intervention is None:
            continue
        out.append(
            {
                "participant_id": participant.participant_id,
                "baseline_mean": baseline,
                "intervention_mean": intervention,
                "oriented_delta": _oriented_delta(endpoint, baseline, intervention),
                "subgroups": participant.subgroups,
            }
        )
    return out


def _is_per_protocol(participant: Participant) -> bool:
    if participant.status not in {"completed", "complete"}:
        return False
    pct = participant.verified_compliance_pct
    return pct is None or float(pct) >= MIN_PER_PROTOCOL_COMPLIANCE


def _oriented_delta(endpoint: Endpoint, baseline: float, intervention: float) -> float:
    if endpoint.direction == "lower_better":
        return baseline - intervention
    if endpoint.direction == "optimal_range":
        return _distance_from_range(baseline, endpoint.optimal_min, endpoint.optimal_max) - _distance_from_range(
            intervention, endpoint.optimal_min, endpoint.optimal_max
        )
    return intervention - baseline


def _distance_from_range(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is None or high is None or low > high:
        return 0.0
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(numeric)) if numeric else None


def _ci95(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if values.size == 0:
        return None, None
    if values.size == 1:
        v = _round_or_none(float(values[0]))
        return v, v
    sem = stats.sem(values)
    margin = stats.t.ppf(0.975, values.size - 1) * sem
    mean = float(np.mean(values))
    return _round_or_none(mean - margin), _round_or_none(mean + margin)


def _wilcoxon_p(values: np.ndarray) -> Optional[float]:
    if values.size < 2 or np.allclose(values, 0):
        return None
    try:
        return _round_or_none(float(stats.wilcoxon(values, zero_method="wilcox").pvalue), digits=6)
    except ValueError:
        return None


def _paired_t_p(baseline: np.ndarray, intervention: np.ndarray) -> Optional[float]:
    if baseline.size < 2 or intervention.size < 2:
        return None
    try:
        pvalue = float(stats.ttest_rel(intervention, baseline, nan_policy="omit").pvalue)
    except ValueError:
        return None
    return _round_or_none(pvalue, digits=6)


def _cohen_dz(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    sd = float(np.std(values, ddof=1))
    return None if sd == 0.0 else _round_or_none(float(np.mean(values)) / sd)


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    return _round_or_none(float(np.mean(values))) if values.size else None


def _sd_or_none(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    return _round_or_none(float(np.std(values, ddof=1)))


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _waterfall_chart(endpoint: Endpoint, participants: Sequence[Participant]) -> Dict[str, Any]:
    rows = sorted(_participant_deltas(endpoint, participants, per_protocol=False), key=lambda item: item["oriented_delta"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    values = [row["oriented_delta"] for row in rows]
    ax.bar(range(len(values)), values, color=["#b94a48" if value < 0 else "#2f7d59" for value in values])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{endpoint.label} responder waterfall")
    ax.set_ylabel("Preferred-direction delta")
    fig.tight_layout()
    return {"endpoint_key": endpoint.key, "image_base64": _fig_to_base64(fig)}


def _time_series_chart(endpoint: Endpoint, participants: Sequence[Participant]) -> Dict[str, Any]:
    series_rows: List[Dict[str, Any]] = []
    for participant in participants:
        data = participant.endpoints.get(endpoint.key)
        if data is None:
            continue
        for idx, value in enumerate(data.daily_values):
            if value is not None:
                series_rows.append({"day": idx + 1, "value": float(value)})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if series_rows:
        frame = pd.DataFrame(series_rows)
        summary = frame.groupby("day")["value"].agg(["mean", "count", "std"]).reset_index()
        summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"].clip(lower=1))
        summary["ci"] = 1.96 * summary["sem"]
        ax.plot(summary["day"], summary["mean"], color="#24515f", linewidth=2)
        ax.fill_between(summary["day"], summary["mean"] - summary["ci"], summary["mean"] + summary["ci"], color="#8fb6c1", alpha=0.35)
    ax.set_title(f"{endpoint.label} cohort time series")
    ax.set_xlabel("Trial day")
    fig.tight_layout()
    return {"endpoint_key": endpoint.key, "image_base64": _fig_to_base64(fig)}


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _subgroup_analysis(endpoints: Sequence[Endpoint], participants: Sequence[Participant]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        for row in _participant_deltas(endpoint, participants, per_protocol=False):
            for name, value in row["subgroups"].items():
                rows.append({"endpoint_key": endpoint.key, "subgroup": name, "value": value, "delta": row["oriented_delta"]})
    if not rows:
        return {"available": False, "groups": []}
    frame = pd.DataFrame(rows)
    groups = []
    for (endpoint_key, subgroup, value), group in frame.groupby(["endpoint_key", "subgroup", "value"]):
        if len(group) >= 2:
            groups.append(
                {
                    "endpoint_key": endpoint_key,
                    "subgroup": subgroup,
                    "value": value,
                    "n": int(len(group)),
                    "mean_delta": _round_or_none(float(group["delta"].mean())),
                    "ci95": dict(zip(["low", "high"], _ci95(group["delta"].to_numpy(dtype=float)), strict=True)),
                }
            )
    return {"available": bool(groups), "groups": groups}


def _dropout_sensitivity(endpoints: Sequence[Endpoint], participants: Sequence[Participant]) -> Dict[str, Any]:
    results = []
    dropout_count = sum(1 for participant in participants if participant.status not in {"completed", "complete"})
    for endpoint in endpoints:
        itt = np.array([row["oriented_delta"] for row in _participant_deltas(endpoint, participants, False)], dtype=float)
        pp = np.array([row["oriented_delta"] for row in _participant_deltas(endpoint, participants, True)], dtype=float)
        results.append(
            {
                "endpoint_key": endpoint.key,
                "dropout_count": int(dropout_count),
                "itt_mean_delta": _round_or_none(float(np.mean(itt))) if itt.size else None,
                "per_protocol_mean_delta": _round_or_none(float(np.mean(pp))) if pp.size else None,
                "delta_gap": _round_or_none(float(np.mean(pp) - np.mean(itt))) if itt.size and pp.size else None,
                "worst_case_dropout_delta": 0.0 if dropout_count else None,
            }
        )
    return {"items": results}


def _methodology(req: AnalyzeRequest) -> Dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "analysis_sets": {
            "itt": "All enrolled participants with baseline and intervention data for the endpoint.",
            "per_protocol": f"Completed participants with verified compliance at or above {MIN_PER_PROTOCOL_COMPLIANCE:.0f} percent.",
        },
        "statistics": [
            "Participant-level baseline and intervention means are compared as paired deltas.",
            "Cohen's dz is computed from paired deltas.",
            "Paired t-test p values are adjusted with Benjamini-Hochberg false-discovery-rate correction.",
            "Wilcoxon signed-rank p values are Bonferroni-adjusted across endpoints.",
            "Responder rate is the share of participants above the endpoint responder threshold.",
            "Time-series confidence bands use normal 95 percent intervals around daily cohort means.",
        ],
        "concurrent_interventions_count": len(req.concurrent_interventions),
        "adverse_events_count": len(req.adverse_events),
    }


PAGE_W_IN = 8.5
PAGE_H_IN = 11.0
BRAND_BLUE = "#1E7FD2"
BRAND_INK = "#0D1B2A"
BRAND_MUTED = "#5B6677"
BRAND_RULE = "#C5CCD6"
BRAND_PANEL = "#F4F6FA"


def _render_pdf(report: Dict[str, Any], report_hash: str) -> bytes:
    buf = io.BytesIO()
    sns.set_theme(style="whitegrid")
    trial = report["trial"]
    trial_label = trial.get("title", "Sponsored Trial")
    pages: List[Tuple[str, Any]] = [
        ("cover", None),
        ("executive", None),
        ("endpoints", None),
        ("methodology", None),
        ("safety", None),
        ("limitations", None),
    ]
    chart_pages: List[Tuple[str, str, Dict[str, Any]]] = []
    for chart in report["charts"].get("waterfall", []):
        chart_pages.append(("Participant-Level Deltas", "Each bar is one participant's mean change from baseline.", chart))
    for chart in report["charts"].get("time_series", []):
        chart_pages.append(("Cohort Daily Means", "Daily cohort mean with 95% confidence band across the on-protocol window.", chart))
    total_pages = len(pages) + len(chart_pages)

    with PdfPages(buf) as pdf:
        page_num = 1
        _draw_cover_page(pdf, report, report_hash, page_num, total_pages)
        page_num += 1
        _draw_executive_page(pdf, report, trial_label, report_hash, page_num, total_pages)
        page_num += 1
        _draw_endpoint_table_page(pdf, report, trial_label, report_hash, page_num, total_pages)
        page_num += 1
        _draw_section_page(pdf, "Methodology", _methodology_lines(report), trial_label, report_hash, page_num, total_pages)
        page_num += 1
        _draw_section_page(pdf, "Safety And Compliance", _safety_lines(report), trial_label, report_hash, page_num, total_pages)
        page_num += 1
        _draw_section_page(pdf, "Limitations And Citation", _limitations_lines(report), trial_label, report_hash, page_num, total_pages)
        page_num += 1
        for title, caption, chart in chart_pages:
            _draw_chart_page(pdf, title, caption, chart, trial_label, report_hash, page_num, total_pages)
            page_num += 1
    return buf.getvalue()


def _draw_page_chrome(fig: Any, trial_label: str, report_hash: str, page_num: int, total_pages: int) -> None:
    """Header bar + footer rule on every non-cover page."""
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.965, "OUTLIYR", fontsize=10, fontweight="bold", color=BRAND_BLUE, family="DejaVu Sans")
    fig.text(0.16, 0.965, "Sponsored Trial Cohort Report", fontsize=10, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.94, 0.965, _truncate(trial_label, 48), fontsize=9, color=BRAND_MUTED, ha="right", family="DejaVu Sans")
    fig.add_artist(plt.Line2D((0.06, 0.94), (0.945, 0.945), color=BRAND_RULE, lw=0.6, transform=fig.transFigure))
    fig.add_artist(plt.Line2D((0.06, 0.94), (0.045, 0.045), color=BRAND_RULE, lw=0.6, transform=fig.transFigure))
    fig.text(0.06, 0.025, f"Outliyr Confidential / Sponsor Use", fontsize=8, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.50, 0.025, f"Page {page_num} of {total_pages}", fontsize=8, color=BRAND_MUTED, ha="center", family="DejaVu Sans")
    fig.text(0.94, 0.025, f"Report {report_hash[:10]}", fontsize=8, color=BRAND_MUTED, ha="right", family="DejaVu Sans")


def _draw_cover_page(pdf: PdfPages, report: Dict[str, Any], report_hash: str, page_num: int, total_pages: int) -> None:
    trial = report["trial"]
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    fig.patch.set_facecolor("white")
    fig.add_artist(plt.Rectangle((0, 0.86), 1.0, 0.14, color=BRAND_INK, transform=fig.transFigure, zorder=0))
    fig.text(0.06, 0.93, "OUTLIYR", fontsize=22, fontweight="bold", color="white", family="DejaVu Sans")
    fig.text(0.06, 0.895, "SPONSORED TRIAL COHORT REPORT", fontsize=10, color="#A8C8EE", family="DejaVu Sans")
    fig.text(0.06, 0.74, _wrap_for_cover(trial.get("title", "Sponsored Trial"), 28), fontsize=30, fontweight="bold", color=BRAND_INK, family="DejaVu Sans", va="top")
    fig.text(0.06, 0.56, "SPONSOR", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.06, 0.535, trial.get("sponsor_name", "Not Recorded"), fontsize=18, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, 0.49, "PRIMARY ENDPOINT", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
    primary = next((e for e in report.get("endpoint_results", []) if e.get("role") == "primary"), None)
    fig.text(0.06, 0.465, primary["label"] if primary else "Not Recorded", fontsize=13, color=BRAND_INK, family="DejaVu Sans")
    if primary:
        fig.text(0.55, 0.49, "PARTICIPANTS", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
        fig.text(0.55, 0.465, f"{primary['n_itt']} ITT · {primary['n_per_protocol']} Per-Protocol", fontsize=13, color=BRAND_INK, family="DejaVu Sans")
    meta_y = 0.36
    fig.add_artist(plt.Line2D((0.06, 0.94), (meta_y + 0.03, meta_y + 0.03), color=BRAND_RULE, lw=0.6, transform=fig.transFigure))
    fig.text(0.06, meta_y, "Engine", fontsize=8, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.06, meta_y - 0.02, report.get("engine_version", ""), fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.30, meta_y, "Generated", fontsize=8, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.30, meta_y - 0.02, _short_iso(report.get("generated_at", "")), fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.55, meta_y, "Protocol Hash", fontsize=8, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.55, meta_y - 0.02, _truncate(trial.get("protocol_hash") or "Not Recorded", 18), fontsize=10, color=BRAND_INK, family="monospace")
    fig.text(0.78, meta_y, "Report Hash", fontsize=8, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.78, meta_y - 0.02, _truncate(report_hash, 18), fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.add_artist(plt.Line2D((0.06, 0.94), (0.045, 0.045), color=BRAND_RULE, lw=0.6, transform=fig.transFigure))
    fig.text(0.06, 0.025, "Outliyr Confidential / Sponsor Use", fontsize=8, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.50, 0.025, f"Page {page_num} of {total_pages}", fontsize=8, color=BRAND_MUTED, ha="center", family="DejaVu Sans")
    fig.text(0.94, 0.025, "outliyr.com", fontsize=8, color=BRAND_MUTED, ha="right", family="DejaVu Sans")
    pdf.savefig(fig)
    plt.close(fig)


def _draw_executive_page(pdf: PdfPages, report: Dict[str, Any], trial_label: str, report_hash: str, page_num: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    _draw_page_chrome(fig, trial_label, report_hash, page_num, total_pages)
    fig.text(0.06, 0.90, "Executive Summary", fontsize=22, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    primary = next((e for e in report.get("endpoint_results", []) if e.get("role") == "primary"), None)
    if not primary:
        fig.text(0.06, 0.85, "No eligible endpoint data was available for this report.", fontsize=11, color=BRAND_MUTED)
        pdf.savefig(fig)
        plt.close(fig)
        return
    hero_y = 0.78
    fig.add_artist(plt.Rectangle((0.055, hero_y - 0.08), 0.89, 0.10, facecolor=BRAND_PANEL, edgecolor=BRAND_RULE, lw=0.6, transform=fig.transFigure, zorder=0))
    delta = primary["itt_mean_delta"]
    fig.text(0.075, hero_y - 0.005, f"{delta:+.2f}" if isinstance(delta, (int, float)) else str(delta), fontsize=44, fontweight="bold", color=BRAND_BLUE, family="DejaVu Sans")
    fig.text(0.30, hero_y - 0.005, "MEAN ITT DELTA", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
    ci = primary.get("ci95", {})
    fig.text(0.30, hero_y - 0.025, f"95% CI {ci.get('low', '?')} to {ci.get('high', '?')}", fontsize=12, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.30, hero_y - 0.045, f"Cohen's dz {primary.get('cohen_d', '?')}", fontsize=11, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.30, hero_y - 0.062, f"Paired t p {primary.get('paired_t_p', '?')} / BH-FDR {primary.get('bh_fdr_p', '?')}", fontsize=10, color=BRAND_MUTED, family="DejaVu Sans")
    fig.text(0.62, hero_y - 0.005, "RESPONDER RATE", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
    rate = primary.get("responder_rate", 0)
    rate_pct = f"{float(rate) * 100:.0f}%" if isinstance(rate, (int, float)) else str(rate)
    fig.text(0.62, hero_y - 0.030, rate_pct, fontsize=24, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.62, hero_y - 0.055, f"{primary.get('responders', '?')} of {primary.get('n_per_protocol', '?')} per-protocol participants", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
    body_y = 0.65
    fig.text(0.06, body_y, "Primary Endpoint", fontsize=11, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.02, primary["label"], fontsize=14, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.05, f"Cohort: {primary['n_itt']} ITT participants, {primary['n_per_protocol']} per-protocol participants.", fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.07, f"Baseline mean: {primary['baseline_mean']} - Intervention mean: {primary['intervention_mean']}.", fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.09, f"Wilcoxon p {primary.get('wilcoxon_p', '?')} - Bonferroni p {primary.get('bonferroni_p', '?')}.", fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.13, "Interpretation", fontsize=11, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.15, "Positive deltas indicate movement in the endpoint's preferred direction. ITT analysis", fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, body_y - 0.165, "includes every enrolled participant; per-protocol restricts to verified-compliant completers.", fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    secondaries = [e for e in report.get("endpoint_results", []) if e.get("role") != "primary"]
    if secondaries:
        sec_y = body_y - 0.22
        fig.text(0.06, sec_y, f"Secondary Endpoints ({len(secondaries)})", fontsize=11, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
        for i, s in enumerate(secondaries[:4]):
            fig.text(0.06, sec_y - 0.022 - i * 0.022, f"- {s['label']}: delta {s['itt_mean_delta']}, dz {s.get('cohen_d', '?')}, BH-FDR p {s.get('bh_fdr_p', '?')}", fontsize=10, color=BRAND_INK, family="DejaVu Sans")
    pdf.savefig(fig)
    plt.close(fig)


def _draw_endpoint_table_page(pdf: PdfPages, report: Dict[str, Any], trial_label: str, report_hash: str, page_num: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    _draw_page_chrome(fig, trial_label, report_hash, page_num, total_pages)
    fig.text(0.06, 0.90, "Endpoint Results", fontsize=22, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    fig.text(0.06, 0.876, "All endpoints with intent-to-treat means, paired-delta effect sizes, and adjusted p-values.", fontsize=9, color=BRAND_MUTED, family="DejaVu Sans")
    rows = report.get("endpoint_results", [])
    if not rows:
        fig.text(0.06, 0.84, "No endpoint rows were available.", fontsize=11, color=BRAND_MUTED)
        pdf.savefig(fig)
        plt.close(fig)
        return
    headers = ["Endpoint", "Role", "N (ITT/PP)", "Baseline", "Intervention", "Delta", "95% CI", "Cohen's dz", "BH-FDR p", "Bonferroni p"]
    body = []
    for r in rows:
        ci = r.get("ci95", {})
        body.append([
            _truncate(r["label"], 22),
            r.get("role", "").title(),
            f"{r.get('n_itt', '?')} / {r.get('n_per_protocol', '?')}",
            str(r.get("baseline_mean", "?")),
            str(r.get("intervention_mean", "?")),
            str(r.get("itt_mean_delta", "?")),
            f"{ci.get('low', '?')} to {ci.get('high', '?')}",
            str(r.get("cohen_d", "?")),
            str(r.get("bh_fdr_p", "?")),
            str(r.get("bonferroni_p", "?")),
        ])
    ax = fig.add_axes([0.06, 0.30, 0.88, 0.55])
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=headers, loc="upper left", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor(BRAND_INK)
        cell.set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(body) + 1):
        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor("white" if row_idx % 2 else BRAND_PANEL)
            cell.set_edgecolor(BRAND_RULE)
    table.auto_set_column_width(col=list(range(len(headers))))
    pdf.savefig(fig)
    plt.close(fig)


def _draw_section_page(pdf: PdfPages, title: str, lines: Sequence[str], trial_label: str, report_hash: str, page_num: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    _draw_page_chrome(fig, trial_label, report_hash, page_num, total_pages)
    fig.text(0.06, 0.90, title, fontsize=22, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    fig.add_artist(plt.Line2D((0.06, 0.20), (0.885, 0.885), color=BRAND_BLUE, lw=2.5, transform=fig.transFigure))
    body_lines: List[str] = []
    for line in lines:
        if line == "":
            body_lines.append("")
            continue
        body_lines.extend(textwrap.wrap(line, width=88) or [""])
    fig.text(0.06, 0.85, "\n".join(body_lines), va="top", ha="left", fontsize=10.5, color=BRAND_INK, family="DejaVu Sans", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_chart_page(pdf: PdfPages, title: str, caption: str, chart: Dict[str, Any], trial_label: str, report_hash: str, page_num: int, total_pages: int) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    _draw_page_chrome(fig, trial_label, report_hash, page_num, total_pages)
    fig.text(0.06, 0.90, title, fontsize=18, fontweight="bold", color=BRAND_INK, family="DejaVu Sans")
    endpoint_label = chart.get("endpoint_label") or chart.get("label") or ""
    if endpoint_label:
        fig.text(0.06, 0.876, endpoint_label, fontsize=11, color=BRAND_MUTED, family="DejaVu Sans")
    ax = fig.add_axes([0.08, 0.18, 0.84, 0.66])
    ax.axis("off")
    img = plt.imread(io.BytesIO(base64.b64decode(chart["image_base64"])), format="png")
    ax.imshow(img)
    fig.text(0.06, 0.135, caption, fontsize=9, color=BRAND_MUTED, family="DejaVu Sans", style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def _truncate(value: str, max_len: int) -> str:
    if not isinstance(value, str):
        return str(value)
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def _wrap_for_cover(value: str, width: int) -> str:
    if not isinstance(value, str):
        value = str(value)
    return "\n".join(textwrap.wrap(value, width=width) or [value])


def _short_iso(value: str) -> str:
    if not isinstance(value, str) or len(value) < 10:
        return value or ""
    return value[:10]


def _methodology_lines(report: Dict[str, Any]) -> List[str]:
    methodology = report["methodology"]
    analysis_sets = methodology.get("analysis_sets", {})
    lines = [
        f"ITT Set: {analysis_sets.get('itt', '')}",
        f"Per-Protocol Set: {analysis_sets.get('per_protocol', '')}",
        "",
        "Statistical Methods:",
    ]
    lines.extend([f"- {item}" for item in methodology.get("statistics", [])])
    return lines


def _safety_lines(report: Dict[str, Any]) -> List[str]:
    methodology = report["methodology"]
    dropout_items = report.get("dropout_sensitivity", {}).get("items", [])
    lines = [
        f"Concurrent Interventions Count: {methodology.get('concurrent_interventions_count', 0)}.",
        f"Adverse Events Count: {methodology.get('adverse_events_count', 0)}.",
        "",
        "Dropout Sensitivity:",
    ]
    if not dropout_items:
        lines.append("No dropout sensitivity rows were available.")
        return lines
    for item in dropout_items:
        lines.append(
            f"{item['endpoint_key']}: Dropouts {item['dropout_count']}; ITT Delta {item['itt_mean_delta']}; "
            f"Per-Protocol Delta {item['per_protocol_mean_delta']}; Gap {item['delta_gap']}."
        )
    return lines


def _limitations_lines(report: Dict[str, Any]) -> List[str]:
    trial = report["trial"]
    return [
        "This report summarizes an open-label sponsored product trial. It is not a randomized, blinded, placebo-controlled clinical trial.",
        "Results may be affected by selection bias, adherence differences, concurrent interventions, device measurement error, "
        "and regression to the mean.",
        "Sponsor claims should describe this as Outliyr cohort evidence unless a separate regulatory or IRB-reviewed study supports "
        "stronger language.",
        "",
        "Suggested Citation:",
        f"Outliyr. {trial.get('title', 'Sponsored Trial')} Cohort Report. Engine {report['engine_version']}. "
        f"Generated {report['generated_at']}.",
    ]
